package com.compa.opencv;

import static org.opencv.core.Core.rectangle;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.imgproc.Imgproc.equalizeHist;
import static org.opencv.imgproc.Imgproc.getAffineTransform;
import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.warpAffine;
import static org.opencv.highgui.Highgui.*;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

public class FaceUtils {

	public static FaceModel getFaceModel(Mat origin) {
		CascadeClassifier faceCascade = new CascadeClassifier();
		CascadeClassifier eyesCascade = new CascadeClassifier();
		CascadeClassifier eyesGlassesCascade = new CascadeClassifier();
		faceCascade
				.load("F:/Downloads/Browsers/opencv-2.4.11/opencv/build/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml");
		eyesCascade
				.load("F:/Downloads/Browsers/opencv-2.4.11/opencv/build/share/OpenCV/haarcascades/haarcascade_eye.xml");
		eyesGlassesCascade
				.load("F:/Downloads/Browsers/opencv-2.4.11/opencv/build/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

		Mat image = origin.clone();
		if (!image.empty()) {
			MatOfRect faces = new MatOfRect();
			Mat frame_gray = new Mat();
			cvtColor(image, frame_gray, COLOR_BGR2GRAY);
			// imwrite("F:/FaceID/ID/test/gray.png", frame_gray);
			equalizeHist(frame_gray, frame_gray);
			faceCascade.detectMultiScale(frame_gray, faces);
			// , 1.1, 2,
			// 0 | CASCADE_SCALE_IMAGE, new Size(30, 30),
			// new Size(30, 30));

			List<Rect> lstRectFace = faces.toList();
			Rect face0 = lstRectFace.get(0);
			if (lstRectFace.size() == 1) {
				
				Rect _face = new Rect(face0.x, face0.y, face0.width,
						face0.height);
				rectangle(image, new Point(_face.x, _face.y), new Point(_face.x
						+ _face.width, _face.y + _face.height), new Scalar(0,
						255, 0, 0), Math.max(1,
						(int) Math.round(image.cols() / 150)), 8, 0);

				// imwrite("F:/FaceID/ID/test/gray2.png", image);
				Point eyeLeft = new Point();
				Point eyeRight = new Point();
				Mat faceROI = new Mat(frame_gray, face0);// ????
				MatOfRect eyes = new MatOfRect();
				eyesGlassesCascade.detectMultiScale(faceROI, eyes);
				// , 1.1, 2,
				// 0 | CASCADE_SCALE_IMAGE, new Size(20, 20), new Size(20,
				// 20));
				List<Rect> lstRectEyes = eyes.toList();
				if (lstRectEyes.size() != 2) {
					eyes.release();
					eyes = new MatOfRect();
					eyesCascade.detectMultiScale(faceROI, eyes);
					// , 1.1, 2,
					// 0 | CASCADE_SCALE_IMAGE, new Size(20, 20),
					// new Size(20, 20));
					lstRectEyes = eyes.toList();
				}

				int r1, r2;
				if (lstRectEyes.size() == 2) {
					Rect eye0 = lstRectEyes.get(0);
					Rect eye1 = lstRectEyes.get(1);
					eyeLeft = new Point(
							(float) (face0.x + eye0.x + eye0.width * 0.5),
							(float) (face0.y + eye0.y + eye0.height * 0.5));
					eyeRight = new Point(
							(float) (face0.x + eye1.x + eye1.width * 0.5),
							(float) (face0.y + eye1.y + eye1.height * 0.5));
					r1 = (int) Math.round((face0.width + face0.height) * 0.15);
					r2 = (int) Math.round((eye1.width + eye1.height) * 0.15);
				} else if (lstRectEyes.size() == 1) {
					eyeLeft = new Point(
							(float) (face0.x + face0.x + face0.width * 0.5),
							(float) (face0.y + face0.y + face0.height * 0.5));
					eyeRight = new Point(image.cols() / 2.0f,
							image.rows() / 2.0f);
					r1 = (int) Math.round((face0.width + face0.height) * 0.15);
					r2 = r1;
				} else {
					eyeLeft = new Point(image.cols() / 3.0f,
							image.rows() / 2.0f);
					eyeRight = new Point((2.0f * image.cols()) / 3.0f,
							image.rows() / 2.0f);
					r1 = (int) Math.round((face0.width + face0.height) * 0.03);
					r2 = r1;
				}

				if (eyeLeft.x > eyeRight.x) {
					Point tmp = eyeRight;
					eyeRight = eyeLeft;
					eyeLeft = tmp;
				}

				FaceModel faceModel = new FaceModel(eyeLeft, eyeRight);
				return faceModel;
			}
		}

		return null;
	}

	static FaceModel getFace(Mat origin) {

		FaceModel faceModel = getFaceModel(origin);
		if (faceModel == null) {
			System.out.println("Cannot detect eye");
			return null;
		}
		Point left = faceModel.leftEye;
		Point right = faceModel.rightEye;
		Mat image = origin.clone();
		image = cropFace(image, left, right, new Point(0.25f, 0.25f), new Size(
				200, 200));

		if (image != null && image.cols() > 0 && image.rows() > 0) {
			System.out.println("persion_id;" + "number_id;" + "checked" + ";"
					+ left.x + ";" + left.y + ";" + right.x + ";" + right.y);

			faceModel.cropedFace = image;
			return faceModel;
		} else {
			System.out.println("Error treating the image.");
		}

		return null;
	}

	static Mat scaleRotateTranslate(Mat image, double angle, Point center,
			Point new_center, double scale) {
		Mat warp = new Mat();
		Mat rot = new Mat();
		if (new_center.x >= 0) {
			Point srcTri0 = new Point(center.x, center.y);
			Point srcTri1 = new Point(center.x + 1, center.y);
			Point srcTri2 = new Point(center.x, center.y + 1);
			MatOfPoint2f srcTri = new MatOfPoint2f(srcTri0, srcTri1, srcTri2);

			Point dstTri0 = new Point(new_center.x, new_center.y);
			Point dstTri1 = new Point(new_center.x + 1, new_center.y);
			Point dstTri2 = new Point(new_center.x, new_center.y + 1);
			MatOfPoint2f dstTri = new MatOfPoint2f(dstTri0, dstTri1, dstTri2);

			warp = new Mat(image.rows(), image.cols(), image.type());

			Mat warp_mat = getAffineTransform(srcTri, dstTri);
			warpAffine(image, warp, warp_mat, warp.size());
		} else
			warp = image;
		Mat rot_mat = getRotationMatrix2D(center, angle, scale);
		warpAffine(warp, rot, rot_mat, warp.size());
		return rot;
	}

	static double distance(Point p1, Point p2) {
		double dx = p2.x - p1.x;
		double dy = p2.y - p1.y;
		return Math.sqrt(dx * dx + dy * dy);
	}

	static Mat rotateMat(Mat image, double angle) {
		// Rotate
		Mat tbTemp = new Mat();
		int x = image.cols();
		int y = image.rows();
		Mat rot = getRotationMatrix2D(new Point(x / 2.0, y / 2.0), angle, 1);
		int length = Math.max(x, y);
		warpAffine(image, tbTemp, rot, new Size(length, length));
		return tbTemp;
	}

	static Mat cropFace(Mat image, Point eye_left, Point eye_right,
			Point offset_pct, Size dest_sz) {
		try {
			// calculate offsets in original image
			double offset_h = Math.floor((offset_pct.x) * dest_sz.width);
			double offset_v = Math.floor((offset_pct.y) * dest_sz.height);
			// get the direction
			double[] eye_direction = new double[] { eye_right.x - eye_left.x,
					eye_right.y - eye_left.y };
			// calc rotation angle in radians
			double rotation = -Math.atan2(eye_direction[1], eye_direction[0]);
			// distance between them
			double dist = distance(eye_left, eye_right);
			// calculate the reference eye-width
			double reference = dest_sz.width - 2.0 * offset_h;
			// scale factor
			double scale = dist / reference;
			// rotate original around the left eye
			image = scaleRotateTranslate(image, rotation, eye_left, new Point(
					-1, -1), 1.0);

			//imwrite("F:/FaceID/ID/test/crop.png", image);
			// crop the rotated image
			Point crop_xy = new Point(eye_left.x - scale * offset_h, eye_left.y
					- scale * offset_v);
			Point crop_size = new Point(dest_sz.width * scale, dest_sz.height
					* scale);

			if (crop_size.x < 50 || crop_size.y < 50)
				return null;

			image = new Mat(image, new Rect((int) crop_xy.x, (int) crop_xy.y,
					(int) crop_size.x, (int) crop_size.y));

			// imwrite("F:/FaceID/ID/me/bfsize.png", image);
			// resize it
			resize(image, image, dest_sz, 0.0, 0.0, INTER_LINEAR);
			return image;
		} catch (Exception ex) {
			System.out.println(ex);
		}

		return null;
	}

}
