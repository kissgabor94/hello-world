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

//kamera adatai
#define HEIGHT 720
#define WIDTH 1280

//Illesztett térgörbe fokszáma (+1)
#define xdim 3

#define kapuX 5000					//kapu távolsága a kamera középpontjától
#define kameramagassag 3800			//kamera magassága
#define labda_sugar 100				//milimeterben


//pontosságot befolyásoló paraméterek
#define hibatures 5
#define margo_x 120
#define margo_y 30

//Globális változók
Serial* SP;
sl::zed::Camera* zed;
std::mutex MemLock;

const int szoglepes = 7;
float conf_canny = 45;
float conf_hough = 17;
int conf_minrad = 10;
int conf_orange_threshold = 128;
int conf_darab_threshold = 4;

Vec3f labda;
Vec3f hatter;
int pixelhatar = 160;		//maximum mennyi lehet a talált pixel értéke a szürkeárnyalatos képen

bool kilep;
int nehezseg = 1;
sl::zed::InitParams parameters;
double idosav = 400;
double sorostreshold = 3;

string lampa = "z";

vector<Vec3f> KalibraciosMatrix;
deque<double> tippek;
const int MA_ablak = 3;

//többszálasítás
cv::Mat akt_left = cv::Mat(HEIGHT, WIDTH, CV_8UC4);
cv::Mat elozo_left = cv::Mat(HEIGHT, WIDTH, CV_8UC4);
long long akt_ido;
long long elozo_ido;
sl::zed::Mat depth_akt;
cv::Mat depth_elozo = cv::Mat(HEIGHT, WIDTH, CV_32F);
sl::zed::Mat conf_akt;
cv::Mat conf_elozo = cv::Mat(HEIGHT, WIDTH, CV_32F);
sl::zed::Mat xyz_akt;
cv::Mat xyz_elozo = cv::Mat(HEIGHT, WIDTH, CV_32FC4);
sl::zed::Mat left_akt;
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
	akt_ido = zed->getCameraTimestamp() / 1000000;

	xyz_akt = zed->retrieveMeasure(sl::zed::MEASURE::XYZ);

	left_akt = zed->retrieveImage(sl::zed::LEFT);

	conf_akt = zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE);
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
cv::Mat PolinomEgyutthatok(vector<Vec4d> meresek, vector<double> sulyok)
{
	//X * A ~= Y
	//idohatványmátrix - N x xdim
	//Az X mátrixot feltöltjük az idõértékek hatványaival
	cv::Mat X = Mat::ones(meresek.size(), xdim, CV_64F);
	cv::Mat W = Mat::zeros(meresek.size(), meresek.size(), CV_64F);
	cv::Mat vissza; cv::Mat vissza_sulyozva;
	for (int i = 0; i < meresek.size(); i++)
	{
		for (int j = 0; j < xdim; j++)
		{
			for (int k = 0; k < j; k++)
			{
				X.at<double>(i, j) *= (meresek[i][3]);
			}
		}
		W.at<double>(i, i) = sulyok[i];
	}

	//Az Y mátrixot feltöltjük az idõértékekhez tartozó pozícióértékekkel [3]
	cv::Mat Y = Mat(meresek.size(), 3, CV_64F);

	for (int i = 0; i < meresek.size(); i++)
	{
		for (int n = 0; n < 3; n++)
		{
			Y.at<double>(i, n) = meresek[i][n];
		}
	}
	vissza = (((X.t() * X).inv()) * X.t()) * Y;
	vissza_sulyozva = (((X.t()*W * X).inv()) * X.t()) *W* Y;
	for (int i = 0; i < xdim; i++)
	{
		vissza.at<double>(i, 0) = vissza_sulyozva.at<double>(i, 0);
		vissza.at<double>(i, 2) = vissza_sulyozva.at<double>(i, 2);
	}
	//Kiszámoljuk az együtthatómátrixot
	return vissza;
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
	if (xdim == 3)
	{
	//	cout << "a=" << to_string(be[2]) << " b=" << to_string(be[1]) << " c=" << to_string(be[0]) << endl;
		double D = be[1] * be[1] - 4 * be[0] * be[2];
		if (D < 0)
		{
			cout << "negativ D" << endl;
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
		double ido = 0;
		cv::solvePoly(be, ki, 1000);
		int db = 0;
		for (int i = 0; i < ki.size().height; i++)
		{
			db = 0;
			if (ki.at<double>(i, 0)>0 && ki.at<double>(i, 0) > ido)
			{
				for (int j = 0; j < ki.size().height; j++)
				{
					if (ki.at<double>(i, 0) == ki.at<double>(j, 0))
						db++;
				}
				if (db == 1 && ido>ki.at<double>(i, 0))
				{
					ido = ki.at<double>(i, 0);
				}
			}
		}
		ki.~Mat();
		return ido;
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
	//alapértelmezésként egységmátrix beállítása
	KalibraciosMatrix.push_back(Vec3f(	1	,	0	,	0	));
	KalibraciosMatrix.push_back(Vec3f(	0	,	1	,	0	));
	KalibraciosMatrix.push_back(Vec3f(	0	,	0	,	1	));
	//HA van kalibrációs fájl, mátrix betöltése
	string ideig;
	if (ifstream("kalib.txt"))
	{
		ifstream kalib("kalib.txt");
		for (int a = 0; a < 3; a++)
		{
			std::getline(kalib, ideig, ',');
			KalibraciosMatrix[a][0] = stof(ideig);
			std::getline(kalib, ideig, ',');
			KalibraciosMatrix[a][1] = stof(ideig);
			std::getline(kalib, ideig);
			KalibraciosMatrix[a][2] = stof(ideig);
		}
		kalib.close();
		std::cout << "Kalibracios fajl beolvasva" << std::endl;
		return true;
	}
	else
	{
		std::cout << "Nincs kalibracios file!" << endl;
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
			conf_darab_threshold = stoi(ideig);
		
		kalibconf.close();
		std::cout << "Config file betoltve" << endl;
		return true;
	}
	else
	{
		std::cout << "Nincs config file!" << endl;
		return false;
	}
}
bool SzinFajlBetoltese()
{
	string ideig;
	float r, g, b;
	int value = 0;
	if (ifstream("color.txt"))
	{
		ifstream kalibcolor("color.txt");

		std::getline(kalibcolor, ideig, ',');
		r = stof(ideig);
		std::getline(kalibcolor, ideig, ',');
		g = stof(ideig);
		std::getline(kalibcolor, ideig, ',');
		b = stoi(ideig);

		labda = (r, g, b);

		std::getline(kalibcolor, ideig, ',');
		r = stof(ideig);
		std::getline(kalibcolor, ideig, ',');
		g = stof(ideig);
		std::getline(kalibcolor, ideig, ',');
		b = stof(ideig);

		hatter = (r, g, b);

		std::getline(kalibcolor, ideig, ',');
		value = stoi(ideig);
		zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_BRIGHTNESS, value, false);
		std::getline(kalibcolor, ideig, ',');
		value = stoi(ideig);
		zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_CONTRAST, value, false);
		std::getline(kalibcolor, ideig, ',');
		value = stoi(ideig);
		zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_HUE, value, false);
		std::getline(kalibcolor, ideig, ',');
		value = stoi(ideig);
		zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_SATURATION, value, false);
		std::getline(kalibcolor, ideig, ',');
		value = stoi(ideig);
		zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_GAIN, value, false);
		std::getline(kalibcolor, ideig, ',');
		value = stoi(ideig);
		zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_EXPOSURE, value, false);
		std::getline(kalibcolor, ideig, ',');
		value = stoi(ideig);
		if (value == 1)
		{
			zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_EXPOSURE, value, true);
			zed->setCameraSettingsValue(sl::zed::ZEDCamera_settings::ZED_GAIN, value, true);
		}
		std::getline(kalibcolor, ideig);
		value = stoi(ideig);
		pixelhatar = value;

		kalibcolor.close();
		std::cout << "Color file betoltve" << endl;
		return true;
	}
	else
	{
		std::cout << "Nincs color file!" << endl;
		return false;
	}
}
void HelpKiirasa()
{
	std::cout << "h: instrukciok" << endl;
	std::cout << "n: nulla pozicioba allas" << endl;
	std::cout << "t: funkcioteszt" << endl;
	std::cout << "p: pozíció keresése" << endl;
	std::cout << "a: ZED ujracsatlakozas" << endl;
	std::cout << "s: soros ujracsatlakozas" << endl;
	std::cout << "c: kalibracio " << endl;
	std::cout << "d: nehezseg beallitasa (inaktív)" << endl;
	std::cout << "q: kilepes" << endl;
	std::cout << "u: ido merese" << endl;
	std::cout << "v: loves varasa (xyz)" << endl;
	std::cout << "o: kameraparameterek es szinek beallitasa" << endl;
}
String SorosFormatum(double szog)
{
	String kimegy = "s"+to_string(nehezseg);
	double S = szog;
	if (S < -80)
		S = -80;
	if (S > 80)
		S = 80;
	if (S >= 0)
	{
		kimegy += "+";
	}
	else
	{
		kimegy += "-";
		S = 0 - S;
	}
	if (S < 10)
		kimegy += "0";
	kimegy += to_string(int(S));
	kimegy += lampa+"e";
	return kimegy;
}
void filebairas(string mit)
{
	bool kint = true;
	string cel = "";
	int szam = 0;
	while (kint)
	{
		cel = "palya" + to_string(szam) + ".csv";
		if (!ifstream(cel))
		{
			ofstream kiir(cel);
			kiir << mit;
			kint = false;
		}
		else
		{
			szam++;
		}
	}
}
double moving_average()
{
	double vissza = 0;
	for (int i = 0; i < MA_ablak; i++)
	{
		vissza += tippek[i];
	}
	vissza /= MA_ablak;
	return vissza;
}
struct Pont {
	int x;
	int y;
	Pont(int x, int y)
	{
		this->x = x;
		this->y = y;
	}
};
struct PontSugar {
	int x;
	int y;
	int r;
	PontSugar(int x, int y, int r)
	{
		this->x = x;
		this->y = y;
		this->r = r;
	}
};
struct PacaManager {
	int xmin, xmax, ymin, ymax;
	int X, Y, r;
	int szelesseg, magassag, count;
	float josag;
	PacaManager(vector<Pont> paca)
	{
		this->xmin = 2000;
		this->ymin = 2000;
		this->xmax = 0;
		this->ymax = 0;

		this->count = 0;

		for each (Pont p in paca)
		{
			(this->count)++;
			if (p.x < xmin)
				xmin = p.x;
			if (p.y < ymin)
				ymin = p.y;
			if (p.x > xmax)
				xmax = p.x;
			if (p.y > ymax)
				ymax = p.y;
		}
		this->szelesseg = xmax - xmin;
		this->magassag = ymax - ymin;
		this->X = (int)((xmin + xmax) / 2);
		this->Y = (int)((ymin + ymax) / 2);
		float optimalis_terulet = 3.142 * szelesseg * magassag / 4;
		this->r = int(sqrt(szelesseg*magassag / 4));
		this->josag = (min(szelesseg, magassag) / max(szelesseg, magassag)) * (min(optimalis_terulet, float(this->count)) / max(optimalis_terulet, float(this->count)));
	}
};
int ElarasztasRekurzio(cv::Mat kep, Pont pont, cv::Mat& flag, vector<Pont>& paca, uchar NarancsThreshold)
{
	if (pont.x > 0 && pont.y > 0 && pont.x < kep.size().width && pont.y < kep.size().height)
	{
	//	cout << pont.x << "  " << pont.y << endl;
		if (flag.at<uchar>(pont.y, pont.x)!=0)
		{
			int count = 0;
			flag.at<uchar>(pont.y, pont.x) = 0;
			if (kep.at<uchar>(pont.y, pont.x) < NarancsThreshold)
			{
				count++;
				paca.push_back(pont);
				count += ElarasztasRekurzio(kep, Pont(pont.x + 1, pont.y), flag, paca, NarancsThreshold);
				count += ElarasztasRekurzio(kep, Pont(pont.x, pont.y + 1), flag, paca, NarancsThreshold);
				count += ElarasztasRekurzio(kep, Pont(pont.x - 1, pont.y), flag, paca, NarancsThreshold);
				count += ElarasztasRekurzio(kep, Pont(pont.x, pont.y - 1), flag, paca, NarancsThreshold);
			}
			return count;
		}
		else return 0;
	}
	else return 0;
}
vector<vector<Pont>> PacakKeresese(cv::Mat kep, uchar NarancsThreshold, int MeretThreshold)
{
	Mat Flags = Mat::ones(kep.size().height, kep.size().width, CV_8U);
	vector<vector<Pont>> vissza;
	int i, j;
	for (i = 0; i < kep.size().width; i++)
	for (j = 0; j < kep.size().height; j++)
	{
		if (Flags.at<uchar>(j, i))
		{
			Flags.at<uchar>(j, i) = 0;
			if (kep.at<uchar>(j, i) < NarancsThreshold)
			{
				vissza.push_back(vector<Pont>());
				if (ElarasztasRekurzio(kep, Pont(i, j), Flags, vissza[vissza.size()-1], NarancsThreshold) < MeretThreshold)
					vissza.pop_back();
			}
		}
	}
	return vissza;
}
PontSugar TalaltLabda(cv::Mat kep, uchar NarancsThreshold, int MeretThreshold)
{
	vector<vector<Pont>> Pacak = PacakKeresese(kep, NarancsThreshold, MeretThreshold);
	PontSugar legjobb_kor = PontSugar(0, 0, 0);
	if (Pacak.size() > 0)
	{
		float legjobb_josag = 0;
		//cout << "Talalt pacak szama: " << to_string(Pacak.size()) << endl;
		for each (vector<Pont> paca in Pacak)
		{
			PacaManager PM = PacaManager(paca);
			if (PM.josag>legjobb_josag)
			{
				legjobb_josag = PM.josag;
				legjobb_kor = PontSugar(PM.X, PM.Y, PM.r);
			}
		}
	}
	return legjobb_kor;
}
cv::Mat DifiKepzes(cv::Mat kep, Vec3f labdaszin, Vec3f hatterszin)
{
	vector<cv::Mat> savok(3);
	cv::split(kep, savok);
	float min = 0, max = 0;
	Vec3f szindifik = (hatterszin - labdaszin) / 255;
	for (int i = 0; i < 3; i++)
	if (szindifik[i]<0)
		min += szindifik[i] * 255;
	else
		max += szindifik[i] * 255;
	float a = 1 / (max - min);
	cv::Mat Difi = cv::Mat::ones(kep.size().height, kep.size().width, CV_8UC1) * (int)(-min * a);
	cv::scaleAdd(savok[0], szindifik[2] * a, Difi, Difi);
	cv::scaleAdd(savok[1], szindifik[1] * a, Difi, Difi);
	cv::scaleAdd(savok[2], szindifik[0] * a, Difi, Difi);
	return Difi;
}
cv::Vec4d helyzet(cv::Mat bal_kep, cv::Mat xyz, cv::Mat konfidencia, bool kirajzolas, double meretskala, cv::Mat maszk)
{
	cv::Mat savok[4];
	split(xyz,savok);					//X, Y, Z, szín szétválasztása
//	cout << "1" << endl;
	cv::Mat difi = savok[2] - maszk;	//különbségképzés a maszkkal
	cv::Mat munkaKep = cv::Mat(HEIGHT*meretskala, WIDTH*meretskala, CV_32F);
	cv::Mat munkaKonf = cv::Mat(HEIGHT*meretskala, WIDTH*meretskala, CV_32F);
	Size kicsi = munkaKep.size();
//	cout << "2" << endl;
	
	//resize(difi,munkaKep,Size(0, 0), meretskala, meretskala);
	resize(difi, munkaKep, kicsi, 0, 0);
	resize(konfidencia, munkaKonf, kicsi, 0, 0);
	munkaKep = munkaKep.mul(munkaKep/munkaKonf);		//különbségnégyzet
//	munkaKep=munkaKep.

//	cout << "3" << endl;
	int r = 34;
	GaussianBlur(munkaKep, munkaKep, Size(2 * r + 1, 2 * r + 1), 10, 10, BORDER_DEFAULT);

//	cout << "4" << endl;
	double valmin, valmax;
	Point locmin, locmax;
	minMaxLoc(munkaKep, &valmin, &valmax, &locmin, &locmax);
//	cout << "min: " << valmin << " max: " << valmax << "min: " << locmin.x << " " << locmin.y << " max: " <<  locmax.x << " " << locmax.y << endl;
//	cout << "5" << endl;
	r = 20;

	int X = (int)((double)locmax.x / meretskala); //találat helye
	int Y = (int)((double)locmax.y / meretskala);
	cout << "X: " << X << " Y: " << Y << endl;
		
	int x, y, u, v, R = r * r;		//találat környezete futóváltozók
	float maszk_atlag = 0;
	int vizsgalt_pixel_darab = 0;
	Vec4f sulyozott_helyzet = Vec4f(0, 0, 0, 0);
	
	Vec4f sz = Vec4f(0, 0, 0, 0);
	float sulyosszeg = 0;
	
	for (x = max(0, X - r); x < min(WIDTH-1, X + r); x++)
	{
		for (y = max(0, Y - r); y < min(HEIGHT-1, Y + r); y++)
		{
			u = x - X; v = y - Y;
			
			Vec4f localpos = xyz.at<Vec4f>(y, x);
			float localkonf = konfidencia.at<float>(y, x);
			float localmaszk = maszk.at<float>(y,x);
			
			if (u*u + v*v < R && isfinite(localpos[0]) && isfinite(localpos[1]) && isfinite(localpos[2]))
			{
				sulyozott_helyzet += localkonf * localpos;
				maszk_atlag += localkonf*localmaszk; 
				
				sulyosszeg += konfidencia.at<float>(y, x);
				vizsgalt_pixel_darab++;
			}
		}
	}
	if (sulyosszeg != 0)
	{
		sulyozott_helyzet = sulyozott_helyzet / sulyosszeg;
		maszk_atlag = maszk_atlag / sulyosszeg;
	}
	
	float dif = maszk_atlag - sulyozott_helyzet[2];
	cout << "z: " << sulyozott_helyzet[2] << " maszk kulonbseg: " << dif << endl;

	if (sulyozott_helyzet[2] != 0 && dif > 100 /*&& Y<680 <-ezmiez?*/)
	{
		cout << "TALALAT!" << endl;
		cv::circle(bal_kep, Size(X, Y), r, 0, 1, 8, 0);
		cout << "x: " << sulyozott_helyzet[0] << " y: " << sulyozott_helyzet[1] << " z: " << sulyozott_helyzet[2] << " darab: " << vizsgalt_pixel_darab << endl;
	}
	else
	{
		cout << "NINCS TALALAT!" << endl;
		sulyozott_helyzet = Vec4f(0, 0, 0, 0);
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
	float hossz = sqrt(	sulyozott_helyzet[0] * sulyozott_helyzet[0] + 
						sulyozott_helyzet[1] * sulyozott_helyzet[1] + 
						sulyozott_helyzet[2] * sulyozott_helyzet[2]);
	sulyozott_helyzet[0] *= (labda_sugar + hossz) / hossz;
	sulyozott_helyzet[1] *= (labda_sugar + hossz) / hossz;
	sulyozott_helyzet[2] *= (labda_sugar + hossz) / hossz;

	sulyozott_helyzet[2] = kameramagassag - sulyozott_helyzet[2];

	return Vec4d((double)sulyozott_helyzet[0], (double)sulyozott_helyzet[1], (double)sulyozott_helyzet[2], (float)vizsgalt_pixel_darab / sulyosszeg);
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
	s[0] *= (labda_sugar + hossz) / hossz;
	s[1] *= (labda_sugar + hossz) / hossz;
	s[2] *= (labda_sugar + hossz) / hossz;

	s[2] = kameramagassag - s[2];

	return Vec4d((double)s[0], (double)s[1], (double)s[2], (float)cs / c);
}
cv::Vec4d helyzet6(cv::Mat bal_kep, cv::Mat xyz, cv::Mat konfidencia, bool kirajzolas)
{
	Vec4f szinatlag = cv::mean(bal_kep);	//Teljes kép színátlaga
	szinatlag = Vec4f(szinatlag[2], szinatlag[1], szinatlag[0], szinatlag[3]); // RGB-bol BGR-be váltás

	int r = 34;	//Gauss sugara

	cv::Mat konf;	//nem akarunk belematatni a konfidenciaképbe
	resize(konfidencia, konf, Size(0, 0), 0.5, 0.5);
	cv::GaussianBlur(konf, konf, Size(r * 2 + 1, r * 2 + 1), 50, 50, cv::BorderTypes::BORDER_REFLECT_101);

	double valmin, valmax;
	cv::Point posmin, posmax;
	minMaxLoc(konf, &valmin, &valmax, &posmin, &posmax);


	r = 20;

	int X = posmin.x * 2;
	int Y = posmin.y * 2;
	int x, y, u, v, R = r * r;
	Vec4f s = Vec4f(0, 0, 0, 0);
	Vec4f sz = Vec4f(0, 0, 0, 0);
	float c = 0;
	int cs = 0;
	for (x = max(0, X - r); x < min(WIDTH, X + r); x++)
	{
		for (y = max(0, Y - r); y < min(HEIGHT, Y + r); y++)
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
	Vec4f kulonbseg = szinatlag - sz;
	float dif = 0;
	for (int p = 0; p < 3; p++)
		dif += kulonbseg[p] * kulonbseg[p];
	cout << "z: " << s[2] << " szindif: " << dif << endl;

	if (s[2] != 0 && dif > 5000)
	{
		cout << "TALALAT!" << endl;
		cv::circle(bal_kep, posmin, r, 0, 1, 8, 0);
		cout << "x: " << s[0] << " y: " << s[1] << " z: " << s[2] << " darab: " << cs << endl;
	}
	else
	{
		cout << "NINCS TALALAT!" << endl;
		s = Vec4f(0, 0, 0, 0);
	}
	if (kirajzolas) //kirajzolás
	{
		konf -= cv::Mat::ones(konf.size(), CV_32F)*valmin;	//
		konf /= valmax - valmin;							//Kirajzoláshoz normálás

		imshow("konfidencia", konf);
		imshow("talalat", bal_kep);
		waitKey(0);
		cvDestroyWindow("talalat");
		cvDestroyWindow("konfidencia");
	}
	s[2] = kameramagassag - s[2];
	return Vec4d((double)s[0], (double)s[1], (double)s[2], (float)cs / c);
}
cv::Vec4d PozicioKereses(bool difkep, vector<double>& suly, cv::Mat maszk)
{

	Vec4d pozi = helyzet5(elozo_left, xyz_elozo, conf_elozo, difkep, 0.25/*, maszk*/);
//	suly.push_back(pozi[3]);
//	pozi[3] = double(elozo_ido);
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

	labda = (255, 0, 0);
	hatter = (0, 255, 0);
	
	bool ZedKapcsolvaVan = false;
	bool SorosKapcsolvaVan = false;
	char FunkcioValasztas = 'h';
	
	char incomingData[256] = "";
	int dataLength = 255;
	int readResult = 0;
	
	ZedKapcsolvaVan = ZedCsatolas();
	SorosKapcsolvaVan = SorosCsatolas();
	nehezseg = 1;
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
						szog = szoglepes * round(szog / szoglepes);
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
					akt_ido = zed->getCameraTimestamp() / 1000000;

					xyz_akt = zed->retrieveMeasure(sl::zed::MEASURE::XYZ);

					left_akt = zed->retrieveImage(sl::zed::LEFT);

					conf_akt = zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE);

				//	MemLock.lock();
					elozo_ido = akt_ido;
					std::memcpy(elozo_left.data, left_akt.data, WIDTH * HEIGHT * 4);
					std::memcpy(conf_elozo.data, conf_akt.data, WIDTH * HEIGHT * 4);
					std::memcpy(xyz_elozo.data, xyz_akt.data, WIDTH * HEIGHT * 16);
				//	MemLock.unlock();
					vector<double> weights;
					long long kezd = getCPUTickCount();
				//	cout << xyz_elozo.rows << endl;
					Vec4d AktualisPozicio = helyzet5(elozo_left, xyz_elozo, conf_elozo, false, 0.25/*, xyz_mask_z*/);
					long long vege = getCPUTickCount();
					cout << "ido: " << double(vege - kezd) / getTickFrequency() << endl;
					AktualisPozicio[3] = double(zed->getCameraTimestamp() / 1000000);
					std::cout << "Pozíció: X: " << AktualisPozicio[0] << " Y: " << AktualisPozicio[1] << " Z: " << AktualisPozicio[2] << endl; //
					if (AktualisPozicio[2] != (double)kameramagassag)
					{
						cv::imshow("Kamerakep", elozo_left);
						cv::waitKey(0);
						cv::destroyWindow("Kamerakep");
					}
					break;
		}
		case 'd':
		case 'D':
		{
					std::cout << "A nehézség új értéke: ";
					char NehezsegChar = ' ';
					std::cin >> NehezsegChar;
					std::cout << endl;
					switch (NehezsegChar)
					{
					case '1':
						nehezseg = 1;
						break;
					case '2':
						nehezseg = 2;
						break;
					case '3':
						nehezseg = 3;
						break;
					case '4':
						nehezseg = 4;
						break;
					case '5':
						nehezseg = 5;
						break;
					default:
						nehezseg = 1;
						break;
					}
					std::cout << "Nehezseg beallitva " << nehezseg << " ertekre" << endl;
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
						elozo_ido = akt_ido; // adatok átmásolása
						memcpy(elozo_left.data, left_akt.data, WIDTH * HEIGHT * 4);
						memcpy(conf_elozo.data, conf_akt.data, WIDTH * HEIGHT * 4);
						memcpy(xyz_elozo.data, xyz_akt.data, WIDTH * HEIGHT * 16);
						xyz_mask += xyz_elozo;
					}
					xyz_mask /= 10;
					cv::split(xyz_mask, sav);
					memcpy(xyz_mask_z.data, sav[2].data, WIDTH * HEIGHT * 4);
					break;
		}
					/*cv::Mat left(HEIGHT, WIDTH, CV_8UC4);
					cv::Mat conf(HEIGHT, WIDTH, CV_32F);
					cv::Mat depth(HEIGHT, WIDTH, CV_32FC4);

					for (int i = 0; i < 3; i++)
					for (int j = 0; j < 3; j++)
					if (i == j)
						KalibraciosMatrix[i][j] = 1;
					else
						KalibraciosMatrix[i][j] = 0;


					sl::zed::Mat LL;
					sl::zed::Mat depthmap;
					sl::zed::Mat confmap;

					Vec4d pontok[2];
					vector<double> weights;

					long long int ts = -1;

					for (int i = 0; i < 2; i++)
					{
						std::cout << i + 1 << ". pont" << endl;

						do
						{
							while (zed->grab(sl::zed::STANDARD, true, true, true))
							{
								;
							}
							// Retrieve left color image
							LL = zed->retrieveImage(sl::zed::SIDE::LEFT);
							std::memcpy(left.data, LL.data, WIDTH*HEIGHT * 4 * sizeof(uchar));
							// Retrieve depth map
							depthmap = zed->retrieveMeasure(sl::zed::MEASURE::XYZ);
							std::memcpy(depth.data, depthmap.data, WIDTH*HEIGHT * sizeof(float)* 4);
							// Retrieve confidence map
							confmap = zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE);
							std::memcpy(conf.data, confmap.data, WIDTH*HEIGHT * sizeof(float));
							pontok[i] = helyzet(left, depth, conf, false, weights, );

							cv::imshow("Kalibracio", left);
							cout << pontok[i][0] << "  " << pontok[i][1] << "  " << pontok[i][2] << endl;
							cout << "Megfelel? (y)" << endl;
						} while (cv::waitKey(0) != 'y');
						cv::destroyWindow("valami");
						//		delete zed;
					}
					Vec4d vektor = pontok[1] - pontok[0];
					double psi = atan2(vektor[2], vektor[1]);
					double ome = atan2(vektor[0], vektor[2]);
					double co = cos(ome);
					double cp = cos(psi);
					double so = sin(ome);
					double sp = sin(psi);
					KalibraciosMatrix[0][0] = co;
					KalibraciosMatrix[1][0] = sp*so;
					KalibraciosMatrix[2][0] = cp*so;
					KalibraciosMatrix[1][1] = cp;
					KalibraciosMatrix[2][1] = -sp;
					KalibraciosMatrix[0][2] = -so;
					KalibraciosMatrix[1][2] = sp*co;
					KalibraciosMatrix[2][2] = cp*co;

					ofstream ki("kalib.txt");
					for (int a = 0; a < 3; a++)
					{
						ki << to_string(KalibraciosMatrix[a][0]) << "," << to_string(KalibraciosMatrix[a][1]) << "," << to_string(KalibraciosMatrix[a][2]) << std::endl;
					}

					//DESTRUKCIÓ
					left.~Mat();
					conf.~Mat();
					depth.~Mat();
					LL.~Mat();
					depthmap.~Mat();
					confmap.~Mat();

					break;
		}*/
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
						for (int i = 0; i < MA_ablak; i++)
						{
							tippek.push_front(0.0);
						}
						vector<double> weights;
						vector<Vec4d> ErzekeltPoziciok;
						vector<Vec4d> MeroPoziciok;
						cv::Vec4d AktualisPozicio;// = PozicioKereses(false, weights);
						cv::Vec4d elozo = cv::Vec4d(-6000, 0, 0, 0);
						string palya_akt = "ido_akt;poz_X;poz_Y;poz_Z;ETA;EYA;EZA;Ax0;Ay0;Az0;Ax1;Ay1;Az1;Ax2;Ay2;Az2;Ax3;Ay3;Az3\n";
						double sorosmost;
						kilep = false;
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
							elozo_ido = akt_ido; // adatok átmásolása
							memcpy(elozo_left.data, left_akt.data, WIDTH * HEIGHT * 4);
							memcpy(conf_elozo.data, conf_akt.data, WIDTH * HEIGHT * 4);
							memcpy(xyz_elozo.data, xyz_akt.data, WIDTH * HEIGHT * 16);
							xyz_mask += xyz_elozo;
						}
						xyz_mask /= 10;
						cv::split(xyz_mask, sav);
						memcpy(xyz_mask_z.data, sav[2].data, WIDTH * HEIGHT * 4);
						std::thread grab(getpicture);
						grab.join();
						grab.~thread();
						MemLock.lock();
						elozo_ido = akt_ido;
						std::memcpy(elozo_left.data, left_akt.data, WIDTH * HEIGHT * 4);
						std::memcpy(conf_elozo.data, conf_akt.data, WIDTH * HEIGHT * 4);
						std::memcpy(xyz_elozo.data, xyz_akt.data, WIDTH * HEIGHT * 16);
						MemLock.unlock();
						MemLock.lock();
						double START_TIME = double(elozo_ido);
						MemLock.unlock();
						//						double START_TIME = double(zed->getCameraTimestamp()) / 1000000;
						double NOW_TIME = 0;

						double ETA;

						//idohatványmátrix - N x xdim
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
							if (AktualisPozicio[2] < kameramagassag && AktualisPozicio[2]!=0  /*&& ErzekeltPoziciok[ErzekeltPoziciok.size()-1][3]!=double(elozo_ido)*/)
							{
								if (AktualisPozicio[0] > elozo[0]/*&& double(elozo_ido)-START_TIME>0*/)
								{
									
									weights.push_back(AktualisPozicio[3]);
									AktualisPozicio[3] = double(elozo_ido);
									NOW_TIME = AktualisPozicio[3] - START_TIME;
									ErzekeltPoziciok.push_back(AktualisPozicio);
									ErzekeltPoziciok[ErzekeltPoziciok.size() - 1][3] = NOW_TIME;
									if (count < hibatures)
										count++;
									else
										votma = true;

									cout << "HELYES - count: " << count << '/' << hibatures << endl;
									std::cout << '\t' << "t=" << NOW_TIME << " - terpozicio: X= " << AktualisPozicio[0] << " Y= " << AktualisPozicio[1] << " Z= " << AktualisPozicio[2] << endl;
									palya_akt += to_string(NOW_TIME) + ';' + to_string(AktualisPozicio[0]) + ';' + to_string(AktualisPozicio[1]) + ';' + to_string(AktualisPozicio[2]) + ';';
								}
								else
								{
									if (count <= 0)
									{
										if (votma)
										{
											kilep = true;
										}
										else
										{
											START_TIME = double(zed->getCameraTimestamp()) / 1000000.0;
										}
									}
									else
										count--;
									cout << "HIBA: visszamozgas - count:" << count << '/' << hibatures << endl;
									palya_akt += to_string(NOW_TIME) + ";;;;";
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
										kilep = true;
									}
									else
										START_TIME = double(zed->getCameraTimestamp()) / 1000000.0;
								}
								else
									count--;
								cout << "HIBA: nincs talalat - count:" << count << '/' << hibatures << endl;

								palya_akt += to_string(NOW_TIME) + ";;;;";
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
							if (ErzekeltPoziciok.size() == conf_darab_threshold-1)
								{
								int nehezsegvolt = nehezseg;
								nehezseg = 3;
								if (ErzekeltPoziciok[2][1]>ErzekeltPoziciok[0][1])
									{
									tippek.push_back(30);
										String output = SorosFormatum(30);
										char nev[7];
										for (int i = 0; i < 7; i++)
											nev[i] = output[i];
										SP->WriteData(nev, 7);
										cout <<"\t"<< "soros output: 30" << endl;
									}
									else
									{
										tippek.push_back(-30);
										String output = SorosFormatum(-30);
										char nev[7];
										for (int i = 0; i < 7; i++)
											nev[i] = output[i];
										SP->WriteData(nev, 7);
										cout << "soros: -30" << endl;
									}
									nehezseg = nehezsegvolt;
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
								be[0] -= kapuX;
								be[2] /= 2;
								ETA = root(be);
								palya_akt += to_string(ETA) + ';';
								be.~vector();

								cout << '\t' << "ETA: " << ETA << " ; ";

								cv::Vec3d erkezes = polinom(A, ETA);
								cout << "y: " << erkezes[1] << " z: " << erkezes[2] << endl;

								{//fajlba iras adatok
									palya_akt += to_string(erkezes[1]) + ";" + to_string(erkezes[2]) + ";";
									for (int beta = 0; beta < 3; beta++)
									for (int ceta = 0; ceta < 3; ceta++)
										palya_akt += to_string(A.at<double>(beta, ceta)) + ";";
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
								if (/*szogallas!=80 && szogallas!= -80 &&*/ ErzekeltPoziciok.size() >= conf_darab_threshold && abs(tippek[tippek.size()-1]-szogallas)>8)
								{
//								cout << "soros: " << szogallas << endl;
									tippek.push_front(szogallas);
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
							elozo_ido = akt_ido;
							std::memcpy(elozo_left.data, left_akt.data, WIDTH * HEIGHT * 4);
							std::memcpy(conf_elozo.data, conf_akt.data, WIDTH * HEIGHT * 4);
							std::memcpy(xyz_elozo.data, xyz_akt.data, WIDTH * HEIGHT * 16);
							MemLock.unlock();
							if (palya_akt.length() > 0)
							if (palya_akt[palya_akt.length() - 1] != '\n')
								palya_akt += '\n';
							grab.join();
							grab.~thread();
						//	kilep = (GetAsyncKeyState(VK_ESCAPE) & 0x8000);
						}while (!kilep && !(GetAsyncKeyState(VK_ESCAPE) & 0x8000));

						int utolso = ErzekeltPoziciok.size() - 1;
						sebessegx = (ErzekeltPoziciok[utolso][0] - ErzekeltPoziciok[0][0]) / (ErzekeltPoziciok[utolso][3] - ErzekeltPoziciok[0][3]);
						sebessegy = (ErzekeltPoziciok[utolso][1] - ErzekeltPoziciok[0][1]) / (ErzekeltPoziciok[utolso][3] - ErzekeltPoziciok[0][3]);
						sebessegz = (ErzekeltPoziciok[utolso][2] - ErzekeltPoziciok[0][2]) / (ErzekeltPoziciok[utolso][3] - ErzekeltPoziciok[0][3]);
						sebesseg = sqrt(sebessegx*sebessegx + sebessegy*sebessegy + sebessegz*sebessegz);
						cout << endl << endl << "A labda sebessege: " << sebesseg * 3.6 << "km/h" << endl;
						filebairas(palya_akt);
						tippek.clear();
					}
		}
		}
		std::cin >> FunkcioValasztas;

	}
				
	cout << endl << "-------Leallas--------" << endl;
		
	SP->~Serial();
		
	for each (Mat var in KalibraciosMatrix)
	{
		var.~Mat();
	}
	KalibraciosMatrix.~vector();
//	MemLock.~_Mutex_base;
	

	akt_left.~Mat();
	elozo_left.~Mat();
	depth_akt.~Mat();
	depth_elozo.~Mat();
	conf_akt.~Mat();
	conf_elozo.~Mat();
	xyz_akt.~Mat();
	xyz_elozo.~Mat();
	delete zed;
	cout << "Objektumok destrualva" << endl;
	std::cin.ignore();
	return 0;
}
