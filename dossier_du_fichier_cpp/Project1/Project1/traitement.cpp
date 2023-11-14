#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

///////////////  Color Detection  //////////////////////
/*
* 
void getContours(Mat imgDil, Mat img) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		cout << area << endl;
		string objectType;

		if (area > 1000)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			cout << conPoly[i].size() << endl;
			boundRect[i] = boundingRect(conPoly[i]);

			int objCor = (int)conPoly[i].size();

			if (objCor == 3) { objectType = ""; } //Tri
			else if (objCor == 4)
			{
				float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
				cout << aspRatio << endl;
				if (aspRatio > 0.95 && aspRatio < 1.05) { objectType = ""; } //Square
				else { objectType = ""; } //Rect
			}
			else if (objCor > 4) { objectType = ""; } //Circle

			drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
			rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
			putText(img, objectType, { boundRect[i].x,boundRect[i].y - 5 }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);
		}
	}
}


void main() {

	string path = "shape.png";
	Mat img = imread(path);
	Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

	// Preprocessing
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);

	getContours(imgDil, img);

	imshow("Image", img);
	imshow("Image Gray", imgGray);
	imshow("Image Blur", imgBlur);
	imshow("Image Canny", imgCanny);
	imshow("Image Dil", imgDil);

	waitKey(0);

} */





/*
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
	cv::Mat image = cv::imread("me.jpg", cv::IMREAD_GRAYSCALE);
	if (image.empty()) {
		std::cout << "Failed to read the image." << std::endl;
		return -1;
	}

	// Appliquer le filtre de Sobel
	cv::Mat sobel_x, sobel_y;
	cv::Sobel(image, sobel_x, CV_64F, 1, 0, 3);
	cv::Sobel(image, sobel_y, CV_64F, 0, 1, 3);

	// Calculer la magnitude des gradients avec Sobel
	cv::Mat sobel_magnitude;
	cv::sqrt(sobel_x.mul(sobel_x) + sobel_y.mul(sobel_y), sobel_magnitude);

	// Appliquer le filtre de Canny
	cv::Mat canny;
	cv::Canny(image, canny, 100, 200);

	// Appliquer le filtre de Kirsh
	cv::Mat kernel_kirsh = (cv::Mat_<int>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
	cv::Mat kirsh;
	cv::filter2D(image, kirsh, -1, kernel_kirsh);

	// Vérifier les dimensions des images
	if (sobel_magnitude.size() != image.size() || canny.size() != image.size() || kirsh.size() != image.size()) {
		std::cout << "Error: Image dimensions do not match." << std::endl;
		return -1;
	}

	// Convertir le type des images pour correspondre à celui de l'image d'origine
	sobel_magnitude.convertTo(sobel_magnitude, image.type());
	canny.convertTo(canny, image.type());
	kirsh.convertTo(kirsh, image.type());

	// Combinaison des résultats des filtres
	cv::Mat combined = sobel_magnitude.clone();

	// Mesurer le temps d'exécution
	auto start = std::chrono::high_resolution_clock::now();

	cv::bitwise_or(combined, canny, combined);
	cv::bitwise_or(combined, kirsh, combined);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	// Afficher les résultats
	cv::namedWindow("Contours", cv::WINDOW_NORMAL);
	cv::imshow("Contours", combined);
	cv::waitKey(0);
	cv::destroyAllWindows();

	std::cout << "Temps d'exécution : " << duration << " ms" << std::endl;

	return 0;
}*/




#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>


// Mesurer le temps d'exécution
auto start = std::chrono::high_resolution_clock::now();

void getContours(cv::Mat imgDil, cv::Mat& img) {
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(imgDil, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point>> conPoly(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());

	for (size_t i = 0; i < contours.size(); i++) {
		int area = cv::contourArea(contours[i]);
		std::cout << "Area: " << area << std::endl;
		std::string objectType;

		if (area > 1000) {
			float peri = cv::arcLength(contours[i], true);
			cv::approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			int objCor = static_cast<int>(conPoly[i].size());
			std::cout << "Object corners: " << objCor << std::endl;
			boundRect[i] = cv::boundingRect(conPoly[i]);

			if (objCor == 3) {
				objectType = "Triangle";//
			}
			else if (objCor == 4) {
				float aspRatio = static_cast<float>(boundRect[i].width) / static_cast<float>(boundRect[i].height);
				std::cout << "Aspect Ratio: " << aspRatio << std::endl;
				if (aspRatio > 0.95 && aspRatio < 1.05) {
					objectType = "Square";//
				}
				else {
					objectType = "Rectangle"; //
				}
			}
			else if (objCor > 4) {
				objectType = "Circle";//
			}

			cv::drawContours(img, conPoly, static_cast<int>(i), cv::Scalar(255, 0, 255), 2);
			cv::rectangle(img, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0, 255, 0), 5);
			cv::putText(img, objectType, { boundRect[i].x, boundRect[i].y - 5 }, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 69, 255), 2);
		}
	}
}

int main() {
	std::string imageName = "simba.png"; // Nom de l'image avec l'extension
	cv::Mat img0 = cv::imread(imageName);
	cv::Mat img = cv::imread(imageName, cv::IMREAD_GRAYSCALE);

	if (img.empty()) {
		std::cout << "Impossible de lire l'image." << std::endl;
		return -1;
	}

	cv::Mat imgBlur;
	cv::GaussianBlur(img, imgBlur, cv::Size(3, 3), 3, 0);

	cv::Mat sobel_x, sobel_y;
	cv::Sobel(imgBlur, sobel_x, CV_64F, 1, 0);
	cv::Sobel(imgBlur, sobel_y, CV_64F, 0, 1);

	cv::Mat imgSobel;
	cv::convertScaleAbs(sobel_x, sobel_x);
	cv::convertScaleAbs(sobel_y, sobel_y);
	cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, imgSobel);

	cv::Mat imgKirsh;
	cv::Mat kernelKirsh = (cv::Mat_<char>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
	cv::filter2D(imgBlur, imgKirsh, CV_8U, kernelKirsh);

	cv::Mat imgCanny;
	cv::Canny(imgBlur, imgCanny, 25, 75);


	cv::Mat imgCombined;
	cv::bitwise_or(imgSobel, imgKirsh, imgCombined);
	cv::bitwise_or(imgCombined, imgCanny, imgCombined);

	cv::Mat imgDil;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(imgCombined, imgDil, kernel);

	
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	getContours(imgDil, img);
	std::cout << "Temps d'exécution : " << duration << " ms" << std::endl;

	cv::namedWindow("Image zéro", cv::WINDOW_NORMAL);
	cv::imshow("Image zéro", img);

	cv::namedWindow("Image", cv::WINDOW_NORMAL);
	cv::imshow("Image", img);
	cv::namedWindow("Image Blur", cv::WINDOW_NORMAL);
	cv::imshow("Image Blur", imgBlur);
	cv::namedWindow("Image Sobel", cv::WINDOW_NORMAL);
	cv::imshow("Image Sobel", imgSobel);
	cv::namedWindow("Image Kirsh", cv::WINDOW_NORMAL);
	cv::imshow("Image Kirsh", imgKirsh);
	cv::namedWindow("Image Canny", cv::WINDOW_NORMAL);
	cv::imshow("Image Canny", imgCanny);
	cv::namedWindow("Image Combined", cv::WINDOW_NORMAL);
	cv::imshow("Image Combined", imgCombined);
	cv::namedWindow("Image Dil", cv::WINDOW_NORMAL);
	cv::imshow("Image Dil", imgDil);

	cv::waitKey(0);
	cv::destroyAllWindows();

	

	return 0;
}


















	
	
	


	





