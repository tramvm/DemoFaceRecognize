package com.compa.opencv.nativec;

import org.opencv.contrib.FaceRecognizer;

public class FisherFaceRecognizerID extends FaceRecognizer{

	private static native long createFisherFaceRecognizer_1();

	private static native long createFisherFaceRecognizer_1(int num_components);

	private static native long createFisherFaceRecognizer_2(int num_components,
			double threshold);

	public FisherFaceRecognizerID() {
		super(createFisherFaceRecognizer_1());
	}

	public FisherFaceRecognizerID(int num_components) {
		super(createFisherFaceRecognizer_1(num_components));
	}

	public FisherFaceRecognizerID(int num_components, double threshold) {
		super(createFisherFaceRecognizer_2(num_components, threshold));
	}

}
