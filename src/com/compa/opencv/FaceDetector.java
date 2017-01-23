package com.compa.opencv;

import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.imgproc.Imgproc.warpAffine;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetector {
	public static Mat getFace(Mat image) {
		CascadeClassifier faceDetector = new CascadeClassifier(
				"F:/Downloads/Browsers/opencv-2.4.11/opencv/build/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml");

		MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);
        //System.out.println("Detect faces: " + faceDetections.toArray().length);
        for (Rect rect : faceDetections.toArray()) {
            //Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x
        	//         + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));

    		Mat matTempRect = image.submat(rect);
    		return matTempRect;
        }
        return null;
	}
}
