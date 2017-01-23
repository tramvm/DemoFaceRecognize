package com.compa.view;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

public class ImgShow {
	JFrame frame;
	JLabel label;

	public ImgShow() {
		frame = new JFrame();
		label = new JLabel();
		frame.getContentPane().add(label);
	}

	public void show(Mat matFrame) {
		// Imgproc.resize(img, img, new Size(640, 480));
		MatOfByte matOfByte = new MatOfByte();
		Highgui.imencode(".jpg", matFrame, matOfByte);
		byte[] byteArray = matOfByte.toArray();
		BufferedImage bufImage = null;
		try {
			InputStream in = new ByteArrayInputStream(byteArray);
			bufImage = ImageIO.read(in);
			label.setIcon(new ImageIcon(bufImage));
			frame.pack();
			frame.setVisible(true);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
