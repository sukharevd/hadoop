package net.sukharevd.hadoop.nn;

import static net.sukharevd.hadoop.nn.NeuralNetworkUtils.readThetasFromThetasDir;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import net.sukharevd.hadoop.entities.Matrix;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

public class NeuralNetworkReducer extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
    private List<Jama.Matrix> thetas;
    private double alpha;
    private double lambda;
    
    private Matrix thetasWritable = new Matrix();

    @Override
    public void configure(JobConf job) {
        super.configure(job);
        thetas = readThetasFromThetasDir(job);
        alpha = Double.parseDouble(job.get("classification.gradient.discent.alpha"));
        lambda = Double.parseDouble(job.get("classification.regularization.lambda"));
    }

    @Override
    public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter)
            throws IOException {
        Jama.Matrix result = null;
        long counter = 0L;
        double errorRate = 0d;
        double J = 0d;
        while (values.hasNext()) {
            String value = values.next().toString();
            String[] split = value.split("\t");
            Matrix matrix = Matrix.valueOf(split[0]);
            J += Double.parseDouble(split[1]);
            errorRate += Integer.parseInt(split[2]);
            counter += Long.parseLong(split[3]);
            if (result == null) {
                result = new Jama.Matrix(matrix.getItems());
            } else {
                result.plusEquals(new Jama.Matrix(matrix.getItems()));
            }
        }
        J /= counter;
        errorRate /= counter;
        System.out.println(key.toString() + ": " + J + "\t" + errorRate + '\t' + counter + "\t" + Arrays.toString(result.getArray()[0]));
        Jama.Matrix delta = result.times(alpha/counter);
        Jama.Matrix regularization = thetas.get(key.get()).times(lambda/counter);
        thetas.get(key.get()).minusEquals(delta.plus(regularization));
        thetasWritable.setItems(thetas.get(key.get()).getArray());
        output.collect(key, new Text(thetasWritable.toString() + "\t" + J + "\t" + errorRate + "\t" + counter));
    }
}