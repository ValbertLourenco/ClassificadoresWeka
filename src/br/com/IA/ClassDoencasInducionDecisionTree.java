package br.com.IA;

import java.io.FileReader;
import java.util.Random;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;

public class ClassDoencasInducionDecisionTree {

	public static void main(String[] args) throws Exception {
		FileReader leitor = new FileReader("src/Resources/doencas2.arff");
		Instances classDoenca = new Instances(leitor);
		classDoenca.setClassIndex(15);
		classDoenca = classDoenca.resample(new Random());

		Instances baseTeste = classDoenca.testCV(3,0);
		Instances baseTreino = classDoenca.trainCV(3,0);

		Id3 inducionDecisionTree = new Id3();
		inducionDecisionTree.buildClassifier(baseTreino);

		System.out.println("real\tIdt");

		for (int i = 0; i < baseTeste.numInstances(); i++) {
			Instance exemplo = baseTeste.instance(i);
			System.out.print(exemplo.classValue());
			exemplo.setClassMissing();
			double classeIdtree = inducionDecisionTree.classifyInstance(exemplo);
			System.out.println("\t" + classeIdtree);
		}
	}
}
