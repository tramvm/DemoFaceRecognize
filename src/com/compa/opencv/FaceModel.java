package com.compa.opencv;

import org.opencv.core.Mat;
import org.opencv.core.Point;

public class FaceModel {
	public Point leftEye;
	public Point rightEye;
	public Mat cropedFace;

	public FaceModel() {

	}

	public FaceModel(Point leftEye, Point rightEye) {
		this.leftEye = leftEye;
		this.rightEye = rightEye;

	}
}
