package com.compa.opencv;

import static org.opencv.core.Core.*;
import static org.opencv.highgui.Highgui.*;
import static org.opencv.imgproc.Imgproc.*;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import com.compa.opencv.nativec.FisherFaceRecognizerID;
import com.compa.view.ImgShow;

public class DetectGridTableNumber2 {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	static ImgShow imgShowOrigin = new ImgShow();
	static ImgShow imgShowFace = new ImgShow();
	static ImgShow imgShowGrid = new ImgShow();
	
	// Scale sourcec image to this size
	static int sourceWidth = 1366;
	// Scale grid to this size
	static int gridWidth = 1140;
	
	static int numRow = 19;
	static int numCol = 57; // Only by 6 black block 
	
	// Check marked
	//static int squareSize = (gridWidth/numcol);
	//static int halfSize = squareSize/2;
	
	static double threshold = 50.0;
	static int ZOOM_X = 600;
	
	// For train
	static MatOfKeyPoint matKPObj = new MatOfKeyPoint();
	static Mat matDescriptorObj = new Mat();
	static Mat matObj = new Mat();
	static List<KeyPoint> lstKPObj = Collections.emptyList();
	
	static Map<Integer, String> mapIdName = new HashMap<Integer, String>();

	public static void main(String[] args) {
		System.out.println("========= Start Face recognize =========");
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		ttt();

		System.out.println("Done");
	}
	
	static void ttt() {
		Mat templ = imread("F:/grid/c5.jpg");
		Mat origin = imread("F:/grid/8.jpg");
		Size imgSize =  origin.size();
		resize(origin, origin, new Size(sourceWidth, imgSize.height * sourceWidth / imgSize.width), 1.0, 1.0, INTER_CUBIC);
		Mat result = new Mat();
		int matchMethod = TM_SQDIFF;
		Point matchPoint = null;
		Mat sense = origin.clone();
		List<Point> lstPointConner = new ArrayList<>();
		double maxW = 0.0;
		double maxH = 0.0;
		double minW = 10000.0;
		double minH = 10000.0;
		
		for (int k = 0; k < 6; k++) {
			matchTemplate(sense, templ, result, matchMethod);
			Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());

			MinMaxLocResult locResult = minMaxLoc(result, new Mat());
			// rectangle(sense,locResult.maxLoc, new
			// Point(templ.cols(),templ.rows()), new Scalar(255, 0,0), 2, 0, 0);
			if (matchMethod == TM_SQDIFF || matchMethod == TM_SQDIFF_NORMED) {
				matchPoint = locResult.minLoc;
			} else {
				matchPoint = locResult.maxLoc;
			}
			if (matchPoint.x > maxW) {
				maxW = matchPoint.x;
			}
			if (matchPoint.y > maxH) {
				maxH = matchPoint.y;
			}
			
			if (matchPoint.x < minW) {
				minW = matchPoint.x;
			}
			if (matchPoint.y < minH) {
				minH = matchPoint.y;
			}
			
			
			lstPointConner.add(matchPoint);

			// rectangle(origin, locResult.maxLoc, locResult.minLoc, new
			// Scalar(255, 255, 0), 2, 0, 0);
			circle(sense, new Point(matchPoint.x + 5, matchPoint.y + 7), 5, new Scalar(255, 255, 0), 10);
			//circle(sense, new Point(matchPoint.x, matchPoint.y), 5, new Scalar(255, 255, 0), 10);
		}
		System.out.println(maxW + " " + maxH + " " + minW + " " + minH);
		Map<String, Point> mapPoints = mapPointConner(lstPointConner, maxW, maxH, minW, minH);
		// Rebuild list point
//		lstPointConner.clear();
//		lstPointConner.add(mapPoints.get("TL"));
//		lstPointConner.add(mapPoints.get("BL"));
//		lstPointConner.add(mapPoints.get("TM"));
//		lstPointConner.add(mapPoints.get("BM"));
//		lstPointConner.add(mapPoints.get("TR"));
//		lstPointConner.add(mapPoints.get("BR"));
		imwrite("F:/grid/se_out22.jpg", sense);
		// TL->BL: 29, 57
//		circle(sense, mapPoints.get("TL"), 5, new Scalar(255, 0, 0), 10);
//		circle(sense, mapPoints.get("TR"), 5, new Scalar(0,255, 0), 10);
//		circle(sense, mapPoints.get("BR"), 5, new Scalar(0, 0, 255), 10);
//		circle(sense, mapPoints.get("BL"), 5, new Scalar(25, 255, 255), 10);
//		circle(sense, mapPoints.get("BM"), 3, new Scalar(25, 255, 255), 5);
//		circle(sense, mapPoints.get("TM"), 3, new Scalar(0, 0, 255), 5);
		
		Mat originGray = new Mat();
		cvtColor(origin, originGray, COLOR_BGR2GRAY);
		
		Mat element = getStructuringElement(MORPH_RECT, new Size(7, 7), new Point(3, 3));
		dilate(originGray, originGray, element);

		threshold(originGray, originGray, 100, 255, THRESH_BINARY);

		element = getStructuringElement(MORPH_RECT, new Size(7, 7), new Point(3, 3));
		erode(originGray, originGray, element);
		
		double avrgWidth = (mapPoints.get("TR").x - mapPoints.get("TL").x) / (numCol - 1);
		double halfWidth = avrgWidth / 2;
		double avrgWidth2 = (mapPoints.get("BR").x - mapPoints.get("BL").x) / (numCol - 1);
		double halfWidth2 = avrgWidth2 / 2;

		double avrgHeight = (mapPoints.get("BL").y - mapPoints.get("TL").y) / (numRow - 1);
		double halfHeight = avrgHeight / 2;
		double avrgHeight2 = (mapPoints.get("BR").y - mapPoints.get("TR").y) / (numRow - 1);
		double halfHeight2 = avrgHeight2 / 2;
		
		double prePadW = ((mapPoints.get("BL").x - mapPoints.get("TL").x)/(numRow-1));
		double prePadW2 = ((mapPoints.get("BR").x - mapPoints.get("TR").x)/(numRow-1));
		
		double prePadH = ((mapPoints.get("TR").y - mapPoints.get("TL").y)/(numCol-1));
		double prePadH2 = ((mapPoints.get("BR").y - mapPoints.get("BL").y)/(numCol-1));
		
		System.out.println(avrgWidth + " " + avrgHeight);
		
		Map<String, String> mapResult = new HashMap<>();
		for (int j = 1; j<=numRow; j++) {

			String row = "R" + j;
			String col = "";
			double posYMin = (j - 1) * avrgHeight + halfHeight + mapPoints.get("TL").y;
			double posYMax = (j - 1) * avrgHeight2 + halfHeight2 + mapPoints.get("TR").y;
			Point pl=new Point(mapPoints.get("TL").x + (j * prePadW), posYMin);
			Point pr =new Point(mapPoints.get("TR").x + (j * prePadW2) + avrgWidth, posYMax); 

			//line(origin, pl,pr, new Scalar(255, 0, 0));
			
			if (j == 1) {
				System.out.println(row);
				mapResult.put(row, "");
				continue;
			}
			
			
			for (int i = 1; i< numCol -1; i++) {
				if (i == 28 || i == 29 || (j== 19 && i == 1)) {
					continue;
				}
				double posXMin = (i-1) * avrgWidth + halfWidth + mapPoints.get("TL").x;
				double posXMax2 = (i-1) * avrgWidth2 + halfWidth2 + mapPoints.get("BL").x;
				Point pt = new Point(posXMin, mapPoints.get("TL").y + i*prePadH);
				Point pb = new Point(posXMax2, mapPoints.get("BL").y + i*prePadH2 + avrgHeight);
				
				//line(origin, pt, pb ,new Scalar(255, 0, 0));
				
				Point inter = findIntersectionPoint(pl, pr, pt, pb);
				if (inter!= null && originGray.get((int)inter.y, (int)inter.x)[0] == 0){
					col += i + ", ";
					//circle(origin, inter, 2, new Scalar(0, 255, 255), 2);
				}
			}
			mapResult.put(row, col);
			System.out.println(row);
		}
		
		for (int i = 1; i<= numCol; i++) {
			double posXMin = (i-1) * avrgWidth + halfWidth + mapPoints.get("TL").x;
			double posXMax2 = (i-1) * avrgWidth2 + halfWidth2 + mapPoints.get("BL").x;
			
			line(origin, new Point(posXMin, mapPoints.get("TL").y + i*prePadH), new Point(posXMax2, mapPoints.get("BL").y + i*prePadH2 + avrgHeight), new Scalar(255, 0, 0));
		}
		
		//imwrite("F:/grid/se_out.jpg", origin);
		/*
		// Sub grid
		MatOfPoint mOPoint = new MatOfPoint();
		mOPoint.fromList(lstPointConner);
		Rect rectTemp = boundingRect(mOPoint);

		System.out.println("Rect W: " + rectTemp.width + "  H: " + rectTemp.height);


		// Draw rect bounding
		// rectangle(matScene, rectTemp.tl(), rectTemp.br(),
		// new Scalar(0, 255, 0), 3, 8, 0);

		// Rotate
		MatOfPoint2f mTemp = new MatOfPoint2f(mOPoint.toArray());
		RotatedRect temp = Imgproc.minAreaRect(mTemp);
		mOPoint.release();
		mTemp.release();
		System.out.println("Angle: " + temp.angle);
		double angle = temp.angle;
		if (angle <= -45) {
			angle += 90;
		}
		System.out.println("Rotate angle: " + temp.angle);

		Mat tbTemp = new Mat();
		Mat rot = getRotationMatrix2D(temp.center, angle, 1);
		Mat srcGray = new Mat();
		cvtColor(origin, srcGray, COLOR_BGR2GRAY);
		
		warpAffine(srcGray, tbTemp, rot, srcGray.size());

		Mat matTempRect = tbTemp.submat(rectTemp);
		
		//57 x 19 => 1140 x 380
		imgSize = matTempRect.size();
		resize(matTempRect, matTempRect, new Size(gridWidth, imgSize.height * gridWidth / imgSize.width), 1.0, 1.0, INTER_CUBIC);
		
		Mat outRect = matTempRect.clone();
		imwrite("F:/grid/grid_cut.jpg", matTempRect);
		// Apply the dilation, erosion
		 element = getStructuringElement( MORPH_RECT,
                new Size( 5, 5 ),
                new Point( 3, 3 ) );
		dilate( matTempRect, matTempRect, element );
		
		threshold(matTempRect, matTempRect, 100, 150, THRESH_BINARY);
		
		element = getStructuringElement( MORPH_RECT,
                new Size( 5, 5 ),
                new Point( 3, 3 ) );
		erode( matTempRect, matTempRect, element);
		imwrite("F:/grid/erosion.jpg", matTempRect);
		
		System.out.println("Color");
		System.out.println(matTempRect.get(halfSize + squareSize, squareSize +  halfSize)[0]);
		
		for (int j = 1; j<=numRow; j++) {
			String row = "R" + j + ": ";
			if (j == 1) {
				System.out.println(row);
				continue;
			}
			
			int posY = (j-1)*squareSize + halfSize;
			//line(matTempRect, new Point(0, posY), new Point(outRect.width(), posY), new Scalar(255, 0, 0));
			
			for (int i = 1; i< numcol - 2; i++) {
				if (i == 28 || i == 29 || (j== 19 && i == 1)) {
					continue;
				}
				int posX = (i-1) * squareSize + halfSize;
				if (matTempRect.get(posY, posX)[0] == 0){
					row += i + ", ";
					//System.out.println(j+ " x " + i);
					
				}
			}
			System.out.println(row);
			
		}
		
		for (int j = 1; j<=numcol; j++) {
			int posX = (j-1)*squareSize + halfSize;
			line(matTempRect, new Point(posX, 0), new Point(posX, outRect.height()), new Scalar(255, 0, 0));
		}
		
		for (int j = 1; j<=numRow; j++) {

			int posY = (j-1)*squareSize + halfSize;
			line(matTempRect, new Point(0, posY), new Point(outRect.width(), posY), new Scalar(255, 0, 0));
		}
			
		
		imwrite("F:/grid/grid_outrect.jpg", matTempRect);
		*/
	}
	
	// Map points: TL, BL, TR, BT, TM, TB
	static Map<String, Point> mapPointConner(List<Point> lstPointConner, double maxW, double maxH, double minW, double minH) {
		System.out.println("Map point conner");
		Map<String, Point> mapPointConner = new HashMap<>();
		int aroudPic = 100;
		int padding = 0;
		for (Point p : lstPointConner) {
			boolean hitPoint = false;
			// Right
			if (p.x >= maxW - aroudPic) {
				// Bottom right
				if (p.y >= maxH - aroudPic) {
					hitPoint = true;
					System.out.println("BR");
					mapPointConner.put("BR", new Point(p.x + padding, p.y + padding));
				} else {
					hitPoint = true;
					System.out.println("TR");
					mapPointConner.put("TR", new Point(p.x + padding, p.y));
				}
			}
			// Left
			else if (p.x <= minW + aroudPic) {
				// Bottom left
				if (p.y >= maxH - aroudPic) {
					hitPoint = true;
					System.out.println("BL");
					mapPointConner.put("BL", new Point(p.x, p.y + padding));
				} else {
					hitPoint = true;
					System.out.println("TL");
					mapPointConner.put("TL", p);
				}
			}

			// Middle
			if (!hitPoint) {
				if (p.y >= maxH - aroudPic) {
					System.out.println("BM");
					mapPointConner.put("BM", new Point(p.x + padding, p.y + padding));
				} else {
					System.out.println("TM");
					mapPointConner.put("TM", new Point(p.x + padding, p.y));
				}
			}
		}
		
		// More filter
//		double argTempMinY = (mapPointConner.get("TL").y + mapPointConner.get("TM").y + mapPointConner.get("TR").y) / 3;
//		double argTempMaxY = (mapPointConner.get("BL").y + mapPointConner.get("BM").y + mapPointConner.get("BR").y) / 3;
//		double argTempMinX = (mapPointConner.get("TL").x + mapPointConner.get("BL").x)/2;
//		double argTempMidX = (mapPointConner.get("TM").x + mapPointConner.get("BM").x)/2;
//		double argTempMaxX = (mapPointConner.get("TR").x + mapPointConner.get("BR").x)/2;
//		mapPointConner.put("TL", new Point(argTempMinX, argTempMinY));
//		mapPointConner.put("TM", new Point(argTempMidX, argTempMinY));
//		mapPointConner.put("TR", new Point(argTempMaxX, argTempMinY));
//		mapPointConner.put("BL", new Point(argTempMinX, argTempMaxY));
//		mapPointConner.put("BM", new Point(argTempMidX, argTempMaxY));
//		mapPointConner.put("BR", new Point(argTempMaxX, argTempMaxY));
		
		return mapPointConner;
	}
	
	public static double getLineYIntesept(Point p, double slope)
    {
        return p.y - slope * p.x;
    }

    public static Point findIntersectionPoint(Point line1Start, Point line1End, Point line2Start, Point line2End)
    {

    	double slope1 = (line1End.y - line1Start.y) / (line1End.x - line1Start.x);
    	double slope2 = (line2End.y - line2Start.y) / (line2End.x - line2Start.x);

    	double yinter1 = getLineYIntesept(line1Start, slope1);
    	double yinter2 = getLineYIntesept(line2Start, slope2);

        if (slope1 == slope2 && yinter1 != yinter2)
            return null;

        double x = (yinter2 - yinter1) / (slope1 - slope2);

        double y = slope1 * x + yinter1;

        return new Point(x, y);
    }
    
}
