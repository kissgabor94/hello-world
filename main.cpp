#define NOMINMAX
#include <chrono>
#include <iostream>
#include <cv.h>
#include <math.h>
#include <thread>
#include <windows.h>
#include <tchar.h>
#include <mutex>

#include "string"
#include "cstdio"

#include <zed/Camera.hpp>

#include "opencv2\highgui\highgui.hpp"
#include "opencv2\opencv.hpp"

#include "SerialClass.h"

using namespace std;
using namespace cv;
using std::cout;

//Configuration of the camera
#define HEIGHT 720
#define WIDTH 1280

//Degree of the polynomial, used to estimate the trajectory (+1)
#define xdim 3

#define goalX 5000				//Horizontal distance of goal from the center of the camera
#define CameraHeight 3800			//Vertical heigth of camera
#define BallRadius 100				//Millimeters are used throughout the whole project


//Margins of error
#define ErrorMargin 5
#define Marginx 120
#define Marginy 30

//Global variables
Serial* SP;
sl::zed::Camera* zed;
std::mutex MemLock;

const int RotationalResolution = 7;
float conf_canny = 45;
float conf_hough = 17;
int conf_minrad = 10;
int conf_orange_threshold = 128;
int conf_quantity_threshold = 4;

Vec3f Ball;
Vec3f BackGround;
int PixelMargin = 160;		//Intensity threshold of the found pixels

bool Quit;
int Difficulty = 1;
sl::zed::InitParams parameters;
double TimeLine = 400;
double SerialTreshold = 3;

string Lamp = "z";

vector<Vec3f> CalibrationMatrix;
deque<double> Tips;
const int MAWindow = 3;

//Global variables used in the parallel computing
cv::Mat curr_left = cv::Mat(HEIGHT, WIDTH, CV_8UC4);
cv::Mat prev_left = cv::Mat(HEIGHT, WIDTH, CV_8UC4);
long long curr_Time;
long long prev_Time;
sl::zed::Mat depth_curr;
cv::Mat depth_prev = cv::Mat(HEIGHT, WIDTH, CV_32F);
sl::zed::Mat conf_curr;
cv::Mat conf_prev = cv::Mat(HEIGHT, WIDTH, CV_32F);
sl::zed::Mat xyz_curr;
cv::Mat xyz_prev = cv::Mat(HEIGHT, WIDTH, CV_32FC4);
sl::zed::Mat left_curr;
cv::Mat xyz_mask = cv::Mat(HEIGHT, WIDTH, CV_32FC4);
cv::Mat xyz_mask_z = cv::Mat(HEIGHT, WIDTH, CV_32F);

float NarancsKorben(cv::Mat DifKep, float X, float Y, float sugar)
{
	int x, y, r, i, j, s = 0, c = 0, u, v, R;
	x = int(X);
	y = int(Y);
	r = int(sugar);
	R = r * r;
	try
	{
	//	cout << "narancs try indul" << endl;
		for (i = max(0, (x - r)); i < min((x + r)-1, WIDTH-1); i++)
		for (j = max(0, (y - r)); j < min(HEIGHT-1, (y + r)-1); j++)
		{
			u = i - x;
			v = j - y;
			if ((u * u + v * v) < R)
			{
				c++;
				try{
					s += int(DifKep.at<uchar>(i, j));
				}
				catch (...)
				{
					string hiba = "";
					hiba = "u=" + to_string(u) + " v=" + to_string(v) + " i=" + to_string(i) + " j=" + to_string(j) + " x=" + to_string(x) + " y=" + to_string(y) + " r=" + to_string(r);
	//				cout << endl << "ELSZALLAS A NARANCSBAN!!" << endl << hiba << endl<<endl;
				}
			}
		}
	//	cout << "narancs try kozben" << endl;
		float narancs = 0;
		if ((float)c != 0)
			narancs = (float)s / (float)c;
//		cout << "narancs: " << to_string(narancs) << endl;
		return narancs;
	}
	catch (...)
	{
//		cout << "narancs catch" << endl;
		return 255;
	}
}
int IranyIterator(int i)
{
	switch (i)
	{
		if (i <= -7)
			return 0;
		if (i <= -4)
			return -1;
		if (i == -3)
			return 0;
		if (i <= 0)
			return 1;
		if (i == 1)
			return 0;
		if (i <= 4)
			return -1;
		if (i == 5)
			return 0;
		if (i <= 7)
			return 1;
		return 0;
	}
}
void getpicture() 
{
	SetPriorityClass(GetCurrentProcess(), 0x00000100);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	bool HIBA = false;
//	cout << "grab start" << endl;
	while (zed->grab(sl::zed::FILL, true, true, true))
	{
		;
	}

	MemLock.lock();
	curr_Time = zed->getCameraTimestamp() / 1000000;

	xyz_curr = zed->retrieveMeasure(sl::zed::MEASURE::XYZ);

	left_curr = zed->retrieveImage(sl::zed::LEFT);

	conf_curr = zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE);
	MemLock.unlock();
//	cout << "grab end" << endl;

}
Vec3d polinom(cv::Mat A, double t)
{
	Vec3d vissza = Vec3d(0, 0, 0);
	for (int i = A.size().height; i > 0; i--)
	{
		vissza *= t;
		vissza += Vec3d(A.at<double>(i - 1, 0), A.at<double>(i - 1, 1), A.at<double>(i - 1, 2));
	}
	return vissza;
}
cv::Mat PolinomEgyutthatok(vector<Vec4d> Measurements, vector<double> Weights) //Computes the coefficients of the polynomial
{
	//X * A ~= Y - X: timeexponents, A: coefficients, Y: three-dimensional positions in time points defined in X
	//Timeexponent mtrix - N x xdim
	//X matrix is filled with the exponents of the timestamps
	cv::Mat X = Mat::ones(Measurements.size(), xdim, CV_64F);
	cv::Mat W = Mat::zeros(Measurements.size(), Measurements.size(), CV_64F);
	cv::Mat returnValue; cv::Mat weightedReturnValue;
	for (int i = 0; i < Measurements.size(); i++)
	{
		for (int j = 0; j < xdim; j++)
		{
			for (int k = 0; k < j; k++)
			{
				X.at<double>(i, j) *= (Measurements[i][3]);
			}
		}
		W.at<double>(i, i) = Weights[i];
	}

	//Y matrix is filled with position data [3]
	cv::Mat Y = Mat(Measurements.size(), 3, CV_64F);

	for (int i = 0; i < Measurements.size(); i++)
	{
		for (int n = 0; n < 3; n++)
		{
			Y.at<double>(i, n) = Measurements[i][n];
		}
	}
	//Both weighted and unweighted case is computed, disabling the for loop before returns with latter
	//Estimation is based on the method of least squares
	returnValue = (((X.t() * X).inv()) * X.t()) * Y;
	weightedReturnValue = (((X.t()*W * X).inv()) * X.t()) *W* Y;
	for (int i = 0; i < xdim; i++)
	{
		returnValue.at<double>(i, 0) = weightedReturnValue.at<double>(i, 0);
		returnValue.at<double>(i, 2) = weightedReturnValue.at<double>(i, 2);
	}
	return returnValue;
}
vector<Vec4d> szures(vector<Vec4d> meresek, vector<double> sulyok)
{
	vector<Vec4d> vissza = vector<Vec4d>(meresek);
	Vec3d ertek;
	double kulonbseg;
	double legnagyobb_kulonbseg = 0;
	int kivesz = 0;
	for (int i = 0; i < vissza.size(); i++)
	{
		Vec4d proba = vissza.front();
		vissza.erase(vissza.begin());
		cv::Mat A = PolinomEgyutthatok(vissza,sulyok);
		ertek = polinom(A, proba[3]);
		kulonbseg = (ertek[0] - proba[0]) * (ertek[0] - proba[0]) +
			(ertek[1] - proba[1]) * (ertek[1] - proba[1]) +
			(ertek[2] - proba[2]) * (ertek[2] - proba[2]);
		if (kulonbseg > legnagyobb_kulonbseg)
		{
			legnagyobb_kulonbseg = kulonbseg;
			kivesz = i;
		}
		vissza.push_back(proba);
	}
	cout << "Kivett ertek: " << to_string(legnagyobb_kulonbseg) << " ID: " << to_string(kivesz) << endl;
	vissza.erase(vissza.begin() + kivesz);
	return vissza;
}
double root(vector<double> be)
{
	if (xdim == 3) //Solving second degree polinomial. If result is invalid, 750 is returned as a default estimation
	{
		double D = be[1] * be[1] - 4 * be[0] * be[2];
		if (D < 0)
		{
			return 750;
		}
		double t1, t2;
		D = sqrt(D);
		t1 = -be[1] + D; t1 /= (2 * be[2]);
		t2 = -be[1] - D; t2 /= (2 * be[2]);
		cout << "t1=" << to_string(t1) << " t2="<<to_string(t2) << endl;
		if (t1 >= 0 && (t1 < t2)||(t2<=0))
		{
			return t1;
		}
		if (t2 >= 0)
		{
			return t2;
		}
		return 750;
	}
	else
	{

		cv::Mat ki = cv::Mat::ones(1, 1, CV_64F);
		double Time = 0;
		cv::solvePoly(be, ki, 1000);
		int db = 0;
		for (int i = 0; i < ki.size().height; i++)
		{
			db = 0;
			if (ki.at<double>(i, 0)>0 && ki.at<double>(i, 0) > Time)
			{
				for (int j = 0; j < ki.size().height; j++)
				{
					if (ki.at<double>(i, 0) == ki.at<double>(j, 0))
						db++;
				}
				if (db == 1 && Time>ki.at<double>(i, 0))
				{
					Time = ki.at<double>(i, 0);
				}
			}
		}
		ki.~Mat();
		return Time;
	}
}
bool ZedCsatolas()
{
	try
	{
		zed = new sl::zed::Camera(sl::zed::HD720,60);
		zed->init(parameters);
	//	zed->setConfidenceThreshold(75);
		return true;
	}
	catch(...)
	{
		std::cout << "ZED csatlakoztatasa SIKERTELEN" << std::endl;
		return false;
	}
}
bool SorosCsatolas()
{
	try
	{
		SP = new Serial("COM1");
		std::cout << "Soros port csatlakoztatva" << std::endl;
		return true;
	}
	catch(...)
	{
		std::cout << "Soros port csatlakoztatasa SIKERTELEN" << std::endl;
		return false;
	}
}
bool KalibraciosFajlBetoltese()
{
	//Set identity matrix as a default
	CalibrationMatrix.push_back(Vec3f(	1	,	0	,	0	));
	CalibrationMatrix.push_back(Vec3f(	0	,	1	,	0	));
	CalibrationMatrix.push_back(Vec3f(	0	,	0	,	1	));
	//If there is a calibration file, load it
	string ideig;
	if (ifstream("kalib.txt"))
	{
		ifstream kalib("kalib.txt");
		for (int a = 0; a < 3; a++)
		{
			std::getline(kalib, ideig, ',');
			CalibrationMatrix[a][0] = stof(ideig);
			std::getline(kalib, ideig, ',');
			CalibrationMatrix[a][1] = stof(ideig);
			std::getline(kalib, ideig);
			CalibrationMatrix[a][2] = stof(ideig);
		}
		kalib.close();
		std::cout << "Calibration file loaded" << std::endl;
		return true;
	}
	else
	{
		std::cout << "No calibration file found!" << endl;
		return false;
	}
}
bool KonfiguraciosFajlBetoltese()
{
	string ideig;
	if (ifstream("config.txt"))
	{
		ifstream kalibconf("config.txt");
		
			std::getline(kalibconf, ideig, ',');
			conf_canny = stof(ideig);
			std::getline(kalibconf, ideig, ',');
			conf_hough = stof(ideig);
			std::getline(kalibconf, ideig, ',');
			conf_minrad = stoi(ideig);
			std::getline(kalibconf, ideig, ',');
			conf_orange_threshold = stoi(ideig);
			std::getline(kalibconf, ideig);
			conf_quantity_threshold = stoi(ideig);
		
		kalibconf.close();
		std::cout << "Configuration file loaded" << endl;
		return true;
	}
	else
	{
		std::cout << "No configuration file found!" << endl;
		return false;
	}
}

void HelpKiirasa()
{
	std::cout << "h: Print instructions" << endl;
	std::cout << "n: Set goalkeeper to zero position" << endl;
	std::cout << "p: Run position search on a single image" << endl;
	std::cout << "a: Reconnect ZED" << endl;
	std::cout << "s: Reconnect to serial port" << endl;
	std::cout << "c: Calibration " << endl;
	std::cout << "d: Set difficulty" << endl;
	std::cout << "q: Quit" << endl;
	std::cout << "u: Measure runtime" << endl;
	std::cout << "v: Start main sequence - wait for penalty" << endl;
}
String SorosFormatum(double targetAngle) //Creating a character string for the serial output
{
	String output = "s"+to_string(Difficulty);
	double S = targetAngle;
	if (S < -80)
		S = -80;
	if (S > 80)
		S = 80;
	if (S >= 0)
	{
		output += "+";
	}
	else
	{
		output += "-";
		S = 0 - S;
	}
	if (S < 10)
		output += "0";
	output += to_string(int(S));
	output += Lamp+"e";
	return output;
}
void filebairas(string mit)
{
	bool done = true;
	string dest = "";
	int s = 0;
	while (done)
	{
		dest = "palya" + to_string(s) + ".csv";
		if (!ifstream(dest))
		{
			ofstream kiir(dest);
			kiir << mit;
			done = false;
		}
		else
		{
			s++;
		}
	}
}
double moving_average()
{
	double ret = 0;
	for (int i = 0; i < MAWindow; i++)
	{
		ret += Tips[i];
	}
	ret /= MAWindow;
	return ret;
}
cv::Mat DifiKepzes(cv::Mat Image, Vec3f BallColour, Vec3f BackGroundColour)
{
	vector<cv::Mat> Channels(3);
	cv::split(Image, Channels);
	float min = 0, max = 0;
	Vec3f ColourDifferences = (BackGroundColour - BallColour) / 255;
	for (int i = 0; i < 3; i++)
	if (szindifik[i]<0)
		min += ColourDifferences[i] * 255;
	else
		max += ColourDifferences[i] * 255;
	float a = 1 / (max - min);
	cv::Mat Difi = cv::Mat::ones(Image.size().height, Image.size().width, CV_8UC1) * (int)(-min * a);
	cv::scaleAdd(Channels[0], ColourDifferences[2] * a, Difi, Difi);
	cv::scaleAdd(Channels[1], ColourDifferences[1] * a, Difi, Difi);
	cv::scaleAdd(Channels[2], ColourDifferences[0] * a, Difi, Difi);
	return Difi;
}

cv::Vec4d helyzet5(cv::Mat bal_kep, cv::Mat xyz, cv::Mat konfidencia, bool kirajzolas, double meretskala)
{
	//szinatlag = cv::mean(bal_kep);	//Teljes kép színátlaga
	Scalar alapszin = cv::mean(bal_kep);
	Vec4f szinatlag = Vec4f(alapszin[2], alapszin[1], alapszin[0], alapszin[3]); // RGB-bol BGR-be váltás

	cv::Mat left = cv::Mat(HEIGHT, WIDTH, CV_32FC4);
	cv::Mat difi = cv::Mat(HEIGHT*meretskala, WIDTH*meretskala, CV_32F);
	cv::Mat savok[4];

	bal_kep.convertTo(left, CV_32FC4);

	cv::Mat MunkaKep = cv::Mat(HEIGHT*meretskala, WIDTH*meretskala, CV_32FC4);
	Size kicsi = MunkaKep.size();
	resize(left, MunkaKep, Size(0, 0), meretskala, meretskala);
	MunkaKep /= 255;

	alapszin = cv::mean(MunkaKep);

	cv::Mat konstans = cv::Mat(kicsi.height, kicsi.width, CV_32FC4, alapszin);
	MunkaKep = cv::abs(MunkaKep - konstans);

	double valmin, valmax;
	Point locmin, locmax;

	int r = 34;
	split(MunkaKep, savok);
	difi = savok[0].mul(savok[0]) + savok[1].mul(savok[1]) + savok[2].mul(savok[2]);
	GaussianBlur(difi, difi, Size(2 * r + 1, 2 * r + 1), 10, 10, BORDER_DEFAULT);

	minMaxLoc(difi, &valmin, &valmax, &locmin, &locmax);


	r = 20;

	int X = (int)((double)locmax.x / meretskala);
	int Y = (int)((double)locmax.y / meretskala);
	cout << "X: " << X << " Y: " << Y << endl;
	int x, y, u, v, R = r * r;
	Vec4f s = Vec4f(0, 0, 0, 0);
	Vec4f sz = Vec4f(0, 0, 0, 0);
	float c = 0;
	int cs = 0;
	for (x = max(0, X - r); x < min(WIDTH-1, X + r); x++)
	{
		for (y = max(0, Y - r); y < min(HEIGHT-1, Y + r); y++)
		{
			u = x - X; v = y - Y;
			if (u*u + v*v < R && isfinite(xyz.at<Vec4f>(y, x)[0]) && isfinite(xyz.at<Vec4f>(y, x)[1]) && isfinite(xyz.at<Vec4f>(y, x)[2]))
			{
				s += konfidencia.at<float>(y, x) * xyz.at<Vec4f>(y, x);

				sz[0] += (float)xyz.at<uchar[16]>(y, x)[12];
				sz[1] += (float)xyz.at<uchar[16]>(y, x)[13];
				sz[2] += (float)xyz.at<uchar[16]>(y, x)[14];
				sz[3] += (float)xyz.at<uchar[16]>(y, x)[15];

				c += konfidencia.at<float>(y, x);
				cs++;
			}
		}
	}
	if (c != 0)
		s = s / c;
	if (cs != 0)
		sz = sz / cs;
	Vec4f kulonbseg = szinatlag / 255 - sz;
	float dif = 0;
	for (int p = 0; p < 3; p++)
		dif += kulonbseg[p] * kulonbseg[p];
	cout << "z: " << s[2] << " szindif: " << dif << endl;

	if (s[2] != 0 && dif > 10000 && Y<680)
	{
		cout << "TALALAT!" << endl;
		cv::circle(bal_kep, Size(X, Y), r, 0, 1, 8, 0);
		cout << "x: " << s[0] << " y: " << s[1] << " z: " << s[2] << " darab: " << cs << endl;
	}
	else
	{
		cout << "NINCS TALALAT!" << endl;
		s = Vec4f(0, 0, 0, 0);
	}
	if (kirajzolas)
	{
		difi -= cv::Mat(kicsi.height, kicsi.width, CV_32F, valmin);	//
		difi /= (valmax - valmin);									//Kirajzoláshoz normálás
		imshow("kisdifikep", difi);
		imshow("talalat", bal_kep);
		waitKey(0);
		destroyAllWindows();
	}
	float hossz = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
	s[0] *= (BallRadius + hossz) / hossz;
	s[1] *= (BallRadius + hossz) / hossz;
	s[2] *= (BallRadius + hossz) / hossz;

	s[2] = CameraHeight - s[2];

	return Vec4d((double)s[0], (double)s[1], (double)s[2], (float)cs / c);
}

cv::Vec4d PozicioKereses(bool difkep, vector<double>& suly, cv::Mat maszk)
{

	Vec4d pozi = helyzet5(prev_left, xyz_prev, conf_prev, difkep, 0.25/*, maszk*/);
//	suly.push_back(pozi[3]);
//	pozi[3] = double(prev_Time);
	return pozi;
}


int main(int argc, char** argv)
{
	DWORD dwError, dwPriClass;
	SetPriorityClass(GetCurrentProcess(), 0x000000100);
	dwPriClass = GetPriorityClass(GetCurrentProcess());

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	_tprintf(TEXT("Current priority class is 0x%x\n"), dwPriClass);
	cv::setBreakOnError(true);
	parameters.mode = sl::zed::QUALITY;
	parameters.unit = sl::zed::MILLIMETER;

	Ball = (255, 0, 0);
	BackGround = (0, 255, 0);
	
	bool ZedKapcsolvaVan = false;
	bool SorosKapcsolvaVan = false;
	char FunkcioValasztas = 'h';
	
	char incomingData[256] = "";
	int dataLength = 255;
	int readResult = 0;
	
	ZedKapcsolvaVan = ZedCsatolas();
	SorosKapcsolvaVan = SorosCsatolas();
	Difficulty = 1;
	while (zed->getSelfCalibrationStatus() == 1)
	{
		;
	}

	KalibraciosFajlBetoltese();
	KonfiguraciosFajlBetoltese();
//	SzinFajlBetoltese();
	
	
	if (SorosKapcsolvaVan)
		SP->WriteData("s1+00pe", 7);
	
	while (FunkcioValasztas != 'q' && FunkcioValasztas != 'Q')
	{
		if (SorosKapcsolvaVan)
		{
			if (!SP->IsConnected())
			{
				cout << endl << "SOROS DISCONNECT" << endl << endl;
				SorosKapcsolvaVan = false;
			}
			else
			{
				readResult = SP->ReadData(incomingData, dataLength);
				incomingData[readResult] = 0;
				// IDE JÖN A SOROS BEMENET FELDOLGOZÁSA
			}
		}
		if (ZedKapcsolvaVan)
		{
			if (!zed->isZEDconnected())
			{
				ZedKapcsolvaVan = false;
				std::cout << endl << "ZED DISCONNECT" << endl << endl;
			}
		}
		
		switch (FunkcioValasztas)
		{

		case 'M':
		case 'm':
		{
					while (!(GetAsyncKeyState(VK_ESCAPE) & 0x8000))
					{
						char sorosbe[15];
						//while (!=-1)
						{
							SP->ReadData(sorosbe, 15);
							cout << sorosbe;
						}

						Sleep(500);
					}
					break;
		}
		case 'l':
		case 'L':
		{
					String output = SorosFormatum(60);
					char nev[7];
					for (int i = 0; i < 7; i++)
						nev[i] = output[i];
					SP->WriteData(nev, 7);
					Sleep(500);
					char sorosbe[15];
					//while (!=-1)
					{
						SP->ReadData(sorosbe, 15);
						cout << sorosbe;
					}
					cout << endl;
					break;
		}
		case 'o':
		case 'O':
		{
					SzinFajlBetoltese();
					break;
		}
			//Segítség kiírása
		case 'h':
		case 'H':
		{
					HelpKiirasa();
					break;
		}

		case 'W':
		case 'w':
		{
					double szog = 0;
					string minden;
					string adat;
					string erteky;
					string ertekz;
					string ertekt;
					int max = 0;
					char nev[7];
					ifstream pontok("pontok.csv");
					std::getline(pontok, erteky);
					max = stoi(erteky);
					for (int i = 0; i < max; i++)
					{
						std::getline(pontok, erteky, ';');
						std::getline(pontok, ertekz, ';');
						std::getline(pontok, ertekt);
						szog = atan2(stod(erteky), stod(ertekz)) * 180 / 3.141;
						szog = RotationalResolution * round(szog / RotationalResolution);
						adat = SorosFormatum(szog);
						for (int i = 0; i < 7; i++)
							nev[i] = adat[i];
						SP->WriteData(nev, 7);
						std::cout << '\t' << "soros output: " << adat << "->" << nev << endl;
						Sleep(stoi(ertekt));
					}
					pontok.close();
					break;
					
		}
			//Önálló pozíció
		case 'p':
		case 'P':
		{
				/*	vector<double> weights;
					Vec4d AktualisPozicio = PozicioKereses(true,weights);*/
					while (zed->grab(sl::zed::FILL, true, true, true))
					{
						;
					}
					curr_Time = zed->getCameraTimestamp() / 1000000;

					xyz_curr = zed->retrieveMeasure(sl::zed::MEASURE::XYZ);

					left_curr = zed->retrieveImage(sl::zed::LEFT);

					conf_curr = zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE);

				//	MemLock.lock();
					prev_Time = curr_Time;
					std::memcpy(prev_left.data, left_curr.data, WIDTH * HEIGHT * 4);
					std::memcpy(conf_prev.data, conf_curr.data, WIDTH * HEIGHT * 4);
					std::memcpy(xyz_prev.data, xyz_curr.data, WIDTH * HEIGHT * 16);
				//	MemLock.unlock();
					vector<double> weights;
					long long kezd = getCPUTickCount();
				//	cout << xyz_prev.rows << endl;
					Vec4d AktualisPozicio = helyzet5(prev_left, xyz_prev, conf_prev, false, 0.25/*, xyz_mask_z*/);
					long long vege = getCPUTickCount();
					cout << "Time: " << double(vege - kezd) / getTickFrequency() << endl;
					AktualisPozicio[3] = double(zed->getCameraTimestamp() / 1000000);
					std::cout << "Pozíció: X: " << AktualisPozicio[0] << " Y: " << AktualisPozicio[1] << " Z: " << AktualisPozicio[2] << endl; //
					if (AktualisPozicio[2] != (double)CameraHeight)
					{
						cv::imshow("Kamerakep", prev_left);
						cv::waitKey(0);
						cv::destroyWindow("Kamerakep");
					}
					break;
		}
		case 'd':
		case 'D':
		{
					std::cout << "A nehézség új értéke: ";
					char DifficultyChar = ' ';
					std::cin >> DifficultyChar;
					std::cout << endl;
					switch (DifficultyChar)
					{
					case '1':
						Difficulty = 1;
						break;
					case '2':
						Difficulty = 2;
						break;
					case '3':
						Difficulty = 3;
						break;
					case '4':
						Difficulty = 4;
						break;
					case '5':
						Difficulty = 5;
						break;
					default:
						Difficulty = 1;
						break;
					}
					std::cout << "Difficulty beallitva " << Difficulty << " ertekre" << endl;
					break;
		}
		case 'c':
		case 'C':
		{
					cv::Mat sav[4];
					xyz_mask *= 0; xyz_mask_z *= 0;
					for (int i = 0; i < 10; i++)
					{
						std::thread behuzo_szal(getpicture); //betöltünk egy képet, hogy ne legyen üres az akt
						behuzo_szal.join(); // megvárjuk amíg a szál végez
						behuzo_szal.~thread();//megszüntetjük a szálat
						prev_Time = curr_Time; // adatok átmásolása
						memcpy(prev_left.data, left_curr.data, WIDTH * HEIGHT * 4);
						memcpy(conf_prev.data, conf_curr.data, WIDTH * HEIGHT * 4);
						memcpy(xyz_prev.data, xyz_curr.data, WIDTH * HEIGHT * 16);
						xyz_mask += xyz_prev;
					}
					xyz_mask /= 10;
					cv::split(xyz_mask, sav);
					memcpy(xyz_mask_z.data, sav[2].data, WIDTH * HEIGHT * 4);
					break;
		}
			//nulla pozícióba állás
		case 'n':
		case 'N':
		{
					if (SorosKapcsolvaVan)
						SP->WriteData("s1+00pe", 7);
					else
						std::cout << "Soros port offline. Kapcsolodas az s billentyuvel" << endl;
					break;
		}
			//Életjel
		case 't':
		case 'T':
			std::cout << "ITTVAGYOK" << endl;
			break;

		case 'a':
		case 'A':
			ZedKapcsolvaVan = ZedCsatolas();
			break;

		case 's':
		case 'S':
			SorosKapcsolvaVan = SorosCsatolas();
			break;
		case 'v':
		case 'V':
		{
					if (ZedKapcsolvaVan)
					{
						for (int i = 0; i < MAWindow; i++)
						{
							Tips.push_front(0.0);
						}
						vector<double> weights;
						vector<Vec4d> ErzekeltPoziciok;
						vector<Vec4d> MeroPoziciok;
						cv::Vec4d AktualisPozicio;// = PozicioKereses(false, weights);
						cv::Vec4d elozo = cv::Vec4d(-6000, 0, 0, 0);
						string palya_curr = "Time_curr;poz_X;poz_Y;poz_Z;ETA;EYA;EZA;Ax0;Ay0;Az0;Ax1;Ay1;Az1;Ax2;Ay2;Az2;Ax3;Ay3;Az3\n";
						double sorosmost;
						Quit = false;
						bool firstcall = true;
						bool votma = false;
						double sebesseg;
						double sebessegx;
						double sebessegy;
						double sebessegz;
						//maszk a 10 xyz kép átlagából
						cv::Mat sav[4];
						xyz_mask *= 0;
						for (int i = 0; i < 10; i++)
						{
							std::thread behuzo_szal(getpicture); //betöltünk egy képet, hogy ne legyen üres az akt
							behuzo_szal.join(); // megvárjuk amíg a szál végez
							behuzo_szal.~thread();//megszüntetjük a szálat
							prev_Time = curr_Time; // adatok átmásolása
							memcpy(prev_left.data, left_curr.data, WIDTH * HEIGHT * 4);
							memcpy(conf_prev.data, conf_curr.data, WIDTH * HEIGHT * 4);
							memcpy(xyz_prev.data, xyz_curr.data, WIDTH * HEIGHT * 16);
							xyz_mask += xyz_prev;
						}
						xyz_mask /= 10;
						cv::split(xyz_mask, sav);
						memcpy(xyz_mask_z.data, sav[2].data, WIDTH * HEIGHT * 4);
						std::thread grab(getpicture);
						grab.join();
						grab.~thread();
						MemLock.lock();
						prev_Time = curr_Time;
						std::memcpy(prev_left.data, left_curr.data, WIDTH * HEIGHT * 4);
						std::memcpy(conf_prev.data, conf_curr.data, WIDTH * HEIGHT * 4);
						std::memcpy(xyz_prev.data, xyz_curr.data, WIDTH * HEIGHT * 16);
						MemLock.unlock();
						MemLock.lock();
						double START_TIME = double(prev_Time);
						MemLock.unlock();
						//						double START_TIME = double(zed->getCameraTimestamp()) / 1000000;
						double NOW_TIME = 0;

						double ETA;

						//Timehatványmátrix - N x xdim
						cv::Mat X;
						//pozíciómátrix - N x 3
						cv::Mat Y;
						//együtthatómátrix - xdim x 3
						cv::Mat A;
						//X * A ~= Y

						int count = 0;

						do
						{
							std::thread grab(getpicture);

						//	cout << "pic start" << endl;
							AktualisPozicio = PozicioKereses(false, weights, xyz_mask_z);

						//	cout << "pic end" << endl;
						//	cout << "proba " << AktualisPozicio[0] << " " << AktualisPozicio[1] << " " << AktualisPozicio[2] << endl;
							if (AktualisPozicio[2] < CameraHeight && AktualisPozicio[2]!=0  /*&& ErzekeltPoziciok[ErzekeltPoziciok.size()-1][3]!=double(prev_Time)*/)
							{
								if (AktualisPozicio[0] > elozo[0]/*&& double(prev_Time)-START_TIME>0*/)
								{
									
									weights.push_back(AktualisPozicio[3]);
									AktualisPozicio[3] = double(prev_Time);
									NOW_TIME = AktualisPozicio[3] - START_TIME;
									ErzekeltPoziciok.push_back(AktualisPozicio);
									ErzekeltPoziciok[ErzekeltPoziciok.size() - 1][3] = NOW_TIME;
									if (count < ErrorMargin)
										count++;
									else
										votma = true;

									cout << "HELYES - count: " << count << '/' << ErrorMargin << endl;
									std::cout << '\t' << "t=" << NOW_TIME << " - terpozicio: X= " << AktualisPozicio[0] << " Y= " << AktualisPozicio[1] << " Z= " << AktualisPozicio[2] << endl;
									palya_curr += to_string(NOW_TIME) + ';' + to_string(AktualisPozicio[0]) + ';' + to_string(AktualisPozicio[1]) + ';' + to_string(AktualisPozicio[2]) + ';';
								}
								else
								{
									if (count <= 0)
									{
										if (votma)
										{
											Quit = true;
										}
										else
										{
											START_TIME = double(zed->getCameraTimestamp()) / 1000000.0;
										}
									}
									else
										count--;
									cout << "HIBA: visszamozgas - count:" << count << '/' << ErrorMargin << endl;
									palya_curr += to_string(NOW_TIME) + ";;;;";
								}
								//ha nem érte még el a határt a találatok száma,
								// de visszasüllyedt nullára, akkor töröljük mindet, mert csak valami szarságot talált
								if (!votma && count == 0)
								{
									ErzekeltPoziciok.clear();
								}
						//		elozo = AktualisPozicio;
							}
							else
							{
								if (count <= 0)
								{
									if (votma)
									{
										Quit = true;
									}
									else
										START_TIME = double(zed->getCameraTimestamp()) / 1000000.0;
								}
								else
									count--;
								cout << "HIBA: nincs talalat - count:" << count << '/' << ErrorMargin << endl;

								palya_curr += to_string(NOW_TIME) + ";;;;";
							}

							//----------------------------------------------------------------------------
							if (ErzekeltPoziciok.size() ==2 && firstcall)
							{
								/*if (ErzekeltPoziciok[ErzekeltPoziciok.size() - 1][3] < 0)
								{
									ErzekeltPoziciok.erase(ErzekeltPoziciok.begin());
								}*/
								ErzekeltPoziciok.clear();
								firstcall = false;
							}
							if (ErzekeltPoziciok.size() == conf_quantity_threshold-1)
								{
								int Difficultyvolt = Difficulty;
								Difficulty = 3;
								if (ErzekeltPoziciok[2][1]>ErzekeltPoziciok[0][1])
									{
									Tips.push_back(30);
										String output = SorosFormatum(30);
										char nev[7];
										for (int i = 0; i < 7; i++)
											nev[i] = output[i];
										SP->WriteData(nev, 7);
										cout <<"\t"<< "soros output: 30" << endl;
									}
									else
									{
										Tips.push_back(-30);
										String output = SorosFormatum(-30);
										char nev[7];
										for (int i = 0; i < 7; i++)
											nev[i] = output[i];
										SP->WriteData(nev, 7);
										cout << "soros: -30" << endl;
									}
									Difficulty = Difficultyvolt;
								}
							if (ErzekeltPoziciok.size() >= xdim)
							{
								/*if (ErzekeltPoziciok.size() >= 6)
								{
									MeroPoziciok.clear();
									for (int i = 0; i < ErzekeltPoziciok.size()-2; i++)
									{
										MeroPoziciok.push_back(ErzekeltPoziciok[i + 1]);
									}
								}*/
								A = PolinomEgyutthatok(ErzekeltPoziciok, weights);
								

								vector<double> be;
								for (int i = 0; i < xdim; i++)
									be.push_back(A.at<double>(i, 0));
								be[0] -= goalX;
								be[2] /= 2;
								ETA = root(be);
								palya_curr += to_string(ETA) + ';';
								be.~vector();

								cout << '\t' << "ETA: " << ETA << " ; ";

								cv::Vec3d erkezes = polinom(A, ETA);
								cout << "y: " << erkezes[1] << " z: " << erkezes[2] << endl;

								{//fajlba iras adatok
									palya_curr += to_string(erkezes[1]) + ";" + to_string(erkezes[2]) + ";";
									for (int beta = 0; beta < 3; beta++)
									for (int ceta = 0; ceta < 3; ceta++)
										palya_curr += to_string(A.at<double>(beta, ceta)) + ";";
								}

								double szogallas = atan2(erkezes[1], erkezes[2]) * 180 / 3.141;
						//		sorosmost = moving_average();
							//	sorosmost = sorosmost * 180 / 3.141;
								if (!isfinite(szogallas))
								{
									if (erkezes[1]<0)
										szogallas = 80;
									else szogallas = -80;
								}
								if (szogallas>6) szogallas -= 6;
								if (szogallas<-6) szogallas += 6;
								if (szogallas < -80)
									szogallas = -80;
								if (szogallas > 80)
									szogallas = 80;
								if (/*szogallas!=80 && szogallas!= -80 &&*/ ErzekeltPoziciok.size() >= conf_quantity_threshold && abs(Tips[Tips.size()-1]-szogallas)>8)
								{
//								cout << "soros: " << szogallas << endl;
									Tips.push_front(szogallas);
									if (szogallas > 10)
										szogallas += 4;
									else
									if (szogallas < -10)
										szogallas -= 4;
							//		if (szogallas != 0)
									{
										String output = SorosFormatum(szogallas);
										char nev[7];
										for (int i = 0; i < 7; i++)
											nev[i] = output[i];
										SP->WriteData(nev, 7);
										std::cout << '\t' << "soros output: " << output << "->" << nev << endl;
									}
								}
							}
							MemLock.lock();
							prev_Time = curr_Time;
							std::memcpy(prev_left.data, left_curr.data, WIDTH * HEIGHT * 4);
							std::memcpy(conf_prev.data, conf_curr.data, WIDTH * HEIGHT * 4);
							std::memcpy(xyz_prev.data, xyz_curr.data, WIDTH * HEIGHT * 16);
							MemLock.unlock();
							if (palya_curr.length() > 0)
							if (palya_curr[palya_curr.length() - 1] != '\n')
								palya_curr += '\n';
							grab.join();
							grab.~thread();
						//	Quit = (GetAsyncKeyState(VK_ESCAPE) & 0x8000);
						}while (!Quit && !(GetAsyncKeyState(VK_ESCAPE) & 0x8000));

						int utolso = ErzekeltPoziciok.size() - 1;
						sebessegx = (ErzekeltPoziciok[utolso][0] - ErzekeltPoziciok[0][0]) / (ErzekeltPoziciok[utolso][3] - ErzekeltPoziciok[0][3]);
						sebessegy = (ErzekeltPoziciok[utolso][1] - ErzekeltPoziciok[0][1]) / (ErzekeltPoziciok[utolso][3] - ErzekeltPoziciok[0][3]);
						sebessegz = (ErzekeltPoziciok[utolso][2] - ErzekeltPoziciok[0][2]) / (ErzekeltPoziciok[utolso][3] - ErzekeltPoziciok[0][3]);
						sebesseg = sqrt(sebessegx*sebessegx + sebessegy*sebessegy + sebessegz*sebessegz);
						cout << endl << endl << "A Ball sebessege: " << sebesseg * 3.6 << "km/h" << endl;
						filebairas(palya_curr);
						Tips.clear();
					}
		}
		}
		std::cin >> FunkcioValasztas;

	}
				
	cout << endl << "-------Leallas--------" << endl;
		
	SP->~Serial();
		
	for each (Mat var in CalibrationMatrix)
	{
		var.~Mat();
	}
	CalibrationMatrix.~vector();
//	MemLock.~_Mutex_base;
	

	curr_left.~Mat();
	prev_left.~Mat();
	depth_curr.~Mat();
	depth_prev.~Mat();
	conf_curr.~Mat();
	conf_prev.~Mat();
	xyz_curr.~Mat();
	xyz_prev.~Mat();
	delete zed;
	cout << "Objektumok destrualva" << endl;
	std::cin.ignore();
	return 0;
}
