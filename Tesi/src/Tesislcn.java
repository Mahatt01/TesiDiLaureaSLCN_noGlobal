import java.io.FileReader;
import java.util.List;
import java.util.ArrayList;
import java.io.FileWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import mulan.classifier.transformation.LabelPowerset;
import mulan.evaluation.Evaluation;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.HierarchicalLoss;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.data.MultiLabelInstances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
public class Tesislcn {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        J48 learner1 = new J48();
        SLCN prova=new SLCN(learner1, 1000, 20000);
		MultiLabelInstances dataset=new MultiLabelInstances(arffFilename,xmlFilename);
		prova.build(dataset);
		List<Measure> measures = new ArrayList<Measure>();
        measures.add(new HammingLoss());
        measures.add(new HierarchicalLoss(dataset));
        measures.add(new MicroPrecision(dataset.getNumLabels()));
        measures.add(new MicroRecall(dataset.getNumLabels()));
        measures.add(new MicroFMeasure(dataset.getNumLabels()));
		String unlabeledFilename = Utils.getOption("unlabeled", args);
        FileReader reader = new FileReader(unlabeledFilename);
        Instances unlabeledData = new Instances(reader);

        int numInstances = unlabeledData.numInstances();
        long startTime2 = System.currentTimeMillis();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = unlabeledData.instance(instanceIndex);
            MultiLabelOutput output = prova.makePrediction(instance);
            // do necessary operations with provided prediction output, here just print it out
            System.out.println(output);
		
        }
        long endTime2   = System.currentTimeMillis();
        long totalTime2 = endTime2 - startTime2;
        System.out.println("Prediction Runtime: "+ totalTime2);
    long startTime = System.currentTimeMillis();
    Evaluator eval = new Evaluator();
    Evaluation results = eval.evaluate((MultiLabelLearner)prova, dataset, measures);
	MultipleEvaluation results2 = eval.crossValidate((MultiLabelLearner)prova, dataset, 5);
    long endTime   = System.currentTimeMillis();
    long totalTime = endTime - startTime;
    System.out.println("Runtime: "+ totalTime);
    System.out.println(results.getMeasures().get(4) + "\n");
	System.out.println(results.getMeasures().get(3) + "\n");
	System.out.println(results.getMeasures().get(2) + "\n");
	System.out.println(results.getMeasures().get(1) + "\n");
	System.out.println(results.getMeasures().get(0) + "\n");
	System.out.println(results2);
	}
}
        
