package net.sukharevd.hadoop.nn;

import static net.sukharevd.hadoop.nn.NeuralNetworkUtils.readThetasFromThetasDir;
import static net.sukharevd.hadoop.util.JamaExt.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import net.sukharevd.hadoop.entities.Matrix;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

public class NeuralNetworkMapper extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
    
    private List<Jama.Matrix> thetas;
    private int k;
    private int hl;
    private double lambda;
    
    private IntWritable thetasIdxWritable = new IntWritable();
    private Matrix gradWritable = new Matrix();
    private Text resultText = new Text();
    
    @Override
    public void configure(JobConf job) {
        super.configure(job);
        thetas = readThetasFromThetasDir(job);
        k = job.getInt("classification.k", -1);
        hl = job.getInt("classification.hl", -1);
        lambda = Double.parseDouble(job.get("classification.regularization.lambda"));
    }

    @Override
    public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
        assert (thetas.size() > 0);
        Jama.Matrix example = new Jama.Matrix(Matrix.valueOf(value.toString()).getItems());
        Jama.Matrix x = example.getMatrix(0, example.getRowDimension()-1, 0, example.getColumnDimension()-2);
        int y = (int) example.get(0, example.getColumnDimension()-1);
        Jama.Matrix Y = generateZ(y, k);
        
        Jama.Matrix[] a = new Jama.Matrix[1 + hl + 1];
        Jama.Matrix[] z = new Jama.Matrix[1 + hl + 1];
        z[0] = x;
        a[0] = x;
        for (int i = 1; i <= hl+1; i++) {
            a[i-1] = addOneColumn(a[i-1]);
            z[i] = a[i-1].times(thetas.get(i-1).transpose());
            a[i] = sigmoid(z[i]);
        }
        Jama.Matrix out = a[a.length - 1];
        int predictedClass = indexMax(out);

        double J = -sum(Y.arrayTimes(log(out)).plus(addConstant(Y.uminus(), 1).arrayTimes(log( addConstant(out.uminus(), 1)))));
        for (int i = 0; i < thetas.size(); i++) {
            J += lambda * sum(square(removeFirstColumn(thetas.get(0)))); // div 2
        }

        Jama.Matrix[] delta = new Jama.Matrix[1 + hl]; // for each hidden unit and for output unit.
        delta[hl] = out.minus(Y); // last
        for (int i = hl - 1; i >= 0; i--) {
            delta[i] = delta[i+1].times(removeFirstColumn(thetas.get(i+1))).arrayTimes(sigmoidGradient(z[i+1]));
        }
        Jama.Matrix[] results = new Jama.Matrix[1 + hl];
        for (int i = 0; i <= hl; i++) {
            results[i] = delta[i].transpose().times(a[i]);
            thetasIdxWritable.set(i);
            gradWritable.setItems(results[i].getArray());
            resultText.set(gradWritable.toString() + "\t" + J + "\t" + (/*(predictedClass == 0 && y == 0) || (predictedClass != 0 && y!= 0)*/ /*predictedType.equals(expectedType)*/ predictedClass == y ? 1 : 0) + "\t1");
            output.collect(thetasIdxWritable, resultText);
        }
    }
    
    private Jama.Matrix sigmoidGradient(Jama.Matrix out) {
        double[][] oneArray = new double[out.getRowDimension()][out.getColumnDimension()];
        for (double[] oneVector : oneArray) {
            Arrays.fill(oneVector, 1d);
        }
        Jama.Matrix one = new Jama.Matrix(oneArray);
        return sigmoid(out).arrayTimes(one.minus(sigmoid(out)));
    }

    private Jama.Matrix sigmoid(Jama.Matrix matrix) {
        Jama.Matrix result = matrix.copy();
        for (int i = 0; i < result.getRowDimension(); i++) {
            for (int j = 0; j < result.getColumnDimension(); j++) {
                result.set(i, j, sigmoid(result.get(i, j)));
            }
        }
        return result;
    }

    private double sigmoid(double z) {
        return 1d / (1d + Math.exp(-z));
    }

}
