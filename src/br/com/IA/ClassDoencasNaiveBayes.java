package br.com.IA;

import java.io.FileReader;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

public class ClassDoencasNaiveBayes {

	public static void main(String[] args) throws Exception {
		FileReader leitor = new FileReader("src/Resources/doencas2.arff");
		Instances classDoenca = new Instances(leitor);
		classDoenca.setClassIndex(15);
		classDoenca = classDoenca.resample(new Random());

		Instances baseTeste = classDoenca.testCV(3, 0);
		Instances baseTreino = classDoenca.trainCV(3, 0);

		NaiveBayes naiveBayes = new NaiveBayes();
		naiveBayes.buildClassifier(baseTreino);

		System.out.println("real\tBayes");

		for (int i = 0; i < baseTeste.numInstances(); i++) {
			Instance exemplo = baseTeste.instance(i);
			System.out.print(exemplo.classValue());
			exemplo.setClassMissing();
			double classeNaiveBayes = naiveBayes.classifyInstance(exemplo);
			System.out.println("\t" + classeNaiveBayes);
		}
	}
}
