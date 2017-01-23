package com.compa.opencv;

import static org.opencv.highgui.Highgui.*;
import static org.opencv.imgproc.Imgproc.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.VideoCapture;

import com.compa.opencv.nativec.FisherFaceRecognizerID;
import com.compa.view.ImgShow;

public class PlayMain {

	static ImgShow imgShowOrigin = new ImgShow();
	static ImgShow imgShowFace = new ImgShow();
	static Map<Integer, String> mapIdName = new HashMap<Integer, String>();

	public static void main(String[] args) {
		System.out.println("========= Start Face recognize =========");
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//System.loadLibrary("MyOpencvDLL");
		String id = "test";
		//cameraTrain(id);
		//cameraRecognize();
		// resize200();
//		Mat img1 = imread("F:/grid/RGB.png");
//		int channels = img1.channels();
//		double[] color = img1.get(10, 10);
//		for (int i=0; i< channels; i++) {
//			System.out.println(color[i]);
//		}
		
		System.out.println("Done");
	}

	/**
	 * Capture face and set ID
	 */
	static void cameraTrain(String id) {
		// System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		VideoCapture camera = new VideoCapture(0);

		Mat frame = new Mat();
		camera.read(frame);
		int count = 0;
		int index = 0;
		if (!camera.isOpened()) {
			System.out.println("Error");
		} else {
			while (true) {

				if (camera.read(frame)) {
					count++;
					if (count % 17 == 0) {
						System.out.println("Write image  " + count);
						FaceModel faceModel = FaceUtils.getFace(frame.clone());

						if (faceModel != null) {
							index ++;
							//FaceUtils.getFace(frame.clone());
							imwrite("F:/FaceID/ID/"+id+"/" + index + ".png", faceModel.cropedFace);
							imgShowFace.show(faceModel.cropedFace);
						}
					}
					imgShowOrigin.show(frame);
				}
			}
		}
		camera.release();
	}

	/**
	 * Recognize
	 */
	static void cameraRecognize() {

		VideoCapture camera = new VideoCapture(0);

		Mat frame = new Mat();
		camera.read(frame);
		if (!camera.isOpened()) {
			System.out.println("Error");
		} else {
			FaceRecognizer recognizer = createRecognizer();
			while (true) {

				if (camera.read(frame)) {

					int[] label = { -1, -1, -1, -1, -1 };
					double[] confidence = { -1, -1, -1, -1, -1 };
					FaceModel faceModel = FaceUtils.getFace(frame.clone());
					if (faceModel != null) {
						Mat face =  faceModel.cropedFace;
						// face = imread("F:/FaceID/ID/me/img_20.png");
						imgShowFace.show(face);
						// face = imread("F:/FaceID/ID/mesmile/img_80.png",
						// CV_LOAD_IMAGE_GRAYSCALE );
						Mat resizedFace = new Mat();
						cvtColor(face, face, COLOR_BGR2GRAY);
						resize(face, resizedFace, new Size(200, 200), 1.0, 1.0,
								INTER_CUBIC);

						// imgShowFace.show(resizedFace);
						recognizer.predict(resizedFace, label, confidence);
						// System.out.println(label[0] + "  " + label[1] + "  "
						// + label[2] + "  " + label[3] + "  " + label[4]);
						if (true || confidence[0] > 1000) {
							System.out.println(label[0] + " => "
									+ mapIdName.get(new Integer(label[0])));
							System.out.println(confidence[0]);
						} else {
							System.out.println(confidence[0] + "  " + label[0]
									+ "  New face or cannot detect");
							System.out.println(confidence[1]);
						}
						// System.out.println(confidence[0] + "  " +
						// confidence[1] + "  " + confidence[2] + "  " +
						// confidence[3] + "  " + confidence[4]);
					} else {
						System.out.println("----------");
					}

					imgShowOrigin.show(frame);
				}
			}
		}
		camera.release();
	}

	/**
	 * Training and create recognizer
	 * 
	 * @return
	 */
	static FaceRecognizer createRecognizer() {
		System.out.println("Start training");
		FaceRecognizer recognizer = new FisherFaceRecognizerID();
		List<Mat> src = new ArrayList<Mat>();

		src.add(imread("F:/FaceID/ID/mesmile/1.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/mesmile/2.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/mesmile/3.png", CV_LOAD_IMAGE_GRAYSCALE));

		src.add(imread("F:/FaceID/ID/tri/img_1.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/tri/img_2.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/tri/img_3.png", CV_LOAD_IMAGE_GRAYSCALE));

		src.add(imread("F:/FaceID/ID/me/1.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/me/2.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/me/3.png", CV_LOAD_IMAGE_GRAYSCALE));

		src.add(imread("F:/FaceID/ID/nhan/1.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/nhan/2.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/nhan/3.png", CV_LOAD_IMAGE_GRAYSCALE));

		src.add(imread("F:/FaceID/ID/phu/1.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/phu/2.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/phu/3.png", CV_LOAD_IMAGE_GRAYSCALE));

		src.add(imread("F:/FaceID/ID/no/1.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/no/2.png", CV_LOAD_IMAGE_GRAYSCALE));
		src.add(imread("F:/FaceID/ID/no/3.png", CV_LOAD_IMAGE_GRAYSCALE));

		Mat labels = new Mat(new Size(18, 1), CvType.CV_32SC1);
		labels.put(0, 0, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15,
				15, 15, 16, 16, 16);
		mapIdName.put(new Integer(11), "me smile");
		mapIdName.put(new Integer(12), "tri");
		mapIdName.put(new Integer(13), "me");
		mapIdName.put(new Integer(14), "Nhan");
		mapIdName.put(new Integer(15), "Phu");
		mapIdName.put(new Integer(16), "Other==");
		recognizer.train(src, labels);
		System.out.println("End training");
		return recognizer;
	}

	static void resize200() {
		Mat m = imread("F:/FaceID/ID/tri/img_1.png");
		Mat temp = new Mat();
		// cvtColor(m, m, COLOR_BGR2GRAY);
		resize(m, temp, new Size(200, 200), 1.0, 1.0, INTER_CUBIC);
		imwrite("F:/FaceID/ID/tri/img_1.png", temp);

		m = imread("F:/FaceID/ID/tri/img_2.png");
		// cvtColor(m, m, COLOR_BGR2GRAY);
		resize(m, temp, new Size(200, 200), 1.0, 1.0, INTER_CUBIC);
		imwrite("F:/FaceID/ID/tri/img_2.png", temp);

		m = imread("F:/FaceID/ID/tri/img_3.png");
		// cvtColor(m, m, COLOR_BGR2GRAY);
		resize(m, temp, new Size(200, 200), 1.0, 1.0, INTER_CUBIC);
		imwrite("F:/FaceID/ID/tri/img_3.png", temp);
		/*
		 * m = imread("F:/FaceID/ID/mesmile/img_80.png"); cvtColor(m, m,
		 * COLOR_BGR2GRAY); resize(m, temp, new Size(200, 200), 1.0, 1.0,
		 * INTER_CUBIC); imwrite("F:/FaceID/ID/mesmile/img_80.png", temp);
		 * 
		 * m = imread("F:/FaceID/ID/mesmile/img_120.png"); cvtColor(m, m,
		 * COLOR_BGR2GRAY); resize(m, temp, new Size(200, 200), 1.0, 1.0,
		 * INTER_CUBIC); imwrite("F:/FaceID/ID/mesmile/img_120.png", temp);
		 */
	}
}
