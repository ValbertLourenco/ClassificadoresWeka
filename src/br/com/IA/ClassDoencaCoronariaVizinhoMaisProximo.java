package br.com.IA;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class ClassDoencaCoronariaVizinhoMaisProximo {

	public static void main(String[] args) throws Exception {
		FileReader leitor = new FileReader("src/Resources/risco-doenca-coronaria.arff");
		Instances classDoenca = new Instances(leitor);
		classDoenca.setClassIndex(4);
		classDoenca = classDoenca.resample(new Random());

		Instances baseTeste = classDoenca.testCV(3, 0);
		Instances baseTreino = classDoenca.trainCV(3, 0);

		IBk vizinho = new IBk();
		vizinho.buildClassifier(baseTreino);

		System.out.println("real\tVizinho");

		for (int i = 0; i < baseTeste.numInstances(); i++) {
			Instance exemplo = baseTeste.instance(i);
			System.out.print(exemplo.classValue());
			exemplo.setClassMissing();
			double classeVizinho = vizinho.classifyInstance(exemplo);
			System.out.println("\t" + classeVizinho);
		}
	}
}
