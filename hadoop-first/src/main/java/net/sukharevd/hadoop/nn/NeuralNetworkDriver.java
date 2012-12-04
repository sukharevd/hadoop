package net.sukharevd.hadoop.nn;

import java.io.IOException;

import net.sukharevd.hadoop.entities.Matrix;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class NeuralNetworkDriver extends Configured implements Tool {
    
    public enum NeuralNetworkCounters { SUM_J, REDUCER_COUNTER, PREDICTED_VALUE_CTR, PREDICTED_VALUE }
    
    @Override
    public int run(String[] args) throws Exception {
        Path in = new Path(args[0]);
        Path thetasPath = new Path(args[1]);
        int n = Integer.valueOf(args[2]); // number of features
        int hl = Integer.valueOf(args[3]); // number of hidden layers
        int u = Integer.valueOf(args[4]); // number of hidden units
        int k = Integer.valueOf(args[5]); // number of classes
        double alpha = Double.valueOf(args[6]);
        double lambda = Double.valueOf(args[7]);
        long maxIteration = (args.length >= 9) ? Integer.valueOf(args[8]) : 100;
        System.out.println("n = " + n + "\nhl = " + hl + "\nhu = " + u + "\nk = " + k + "\nalpha = " + alpha + "\nlambda = "
            + lambda + "\nmaxIteration = " + maxIteration);
        Configuration conf = getConf();
        JobConf job = new JobConf(conf, NeuralNetworkDriver.class);
        job.set("thetas.path", thetasPath.suffix("/it0").toString());
        generateInitThetas(conf, job, n, hl, u, k);

        long i;
        for (i = 0; i < maxIteration; i++) {
            conf = getConf();
            job = new JobConf(conf, NeuralNetworkDriver.class);
            job.setLong("classification.iteration", i);
            Path prevout = thetasPath.suffix("/it" + job.get("classification.iteration"));
            Path out = thetasPath.suffix("/it" + (i + 1));
            job.set("thetas.path", prevout.toString());
            job.setInt("classification.k", k);
            job.setInt("classification.hl", hl);
            job.set("classification.gradient.discent.alpha", Double.toString(alpha));
            job.set("classification.regularization.lambda", Double.toString(lambda));
            FileInputFormat.setInputPaths(job, in);
            FileOutputFormat.setOutputPath(job, out);
            job.setJobName("Neural network MapReduce (iteration" + i + ")");
            job.setMapperClass(NeuralNetworkMapper.class);
            job.setCombinerClass(NeuralNetworkCombiner.class);
            job.setReducerClass(NeuralNetworkReducer.class);
            job.setInputFormat(TextInputFormat.class);
            job.setOutputFormat(SequenceFileOutputFormat.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);
//            job.setMaxReduceAttempts(0);
            JobClient.runJob(job);
//            RunningJob runningJob = JobClient.runJob(job);
            //long convergedCounter = runningJob.getCounters().getCounter(Counters.CONVERGED);
            //long allCounter = runningJob.getCounters().getCounter(Counters.ALL);
//            if (convergedCounter == allCounter) {
//                System.out.println("Converged at " + i + "th iteration.");
//                break;
//            } else {
//                System.out.println("Converged " + convergedCounter + " of " + allCounter);
//            }
        }

        return 0;
    }
    
    private void generateInitThetas(Configuration conf, JobConf job, int n, int h, int u, int k) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Path thetasPath = new Path(job.get("thetas.path"));
        if (!fs.exists(thetasPath)) {
            SequenceFile.Writer thetasWriter = SequenceFile.createWriter(fs, conf, thetasPath, IntWritable.class, Text.class);
            thetasWriter.append(new IntWritable(0), new Text(generateThetasI(u, n+1).toString()));
            for (int i = 0; i < h - 1; i++) {
                thetasWriter.append(new IntWritable(i+1), new Text(generateThetasI(u, u+1).toString()));
            }
            thetasWriter.append(new IntWritable(h+1), new Text(generateThetasI(k, u+1).toString()));
            thetasWriter.close();
        }
    }
    
    private Matrix generateThetasI(int outputs, int inputs) {
        double[][] matrix = new double[outputs][inputs];
        for (int i = 0; i < outputs; i++) {
            for (int j = 0; j < inputs; j++) {
                matrix[i][j] = Math.random();                
            }
        }
        
        Matrix matrix2 = new Matrix(matrix);
        System.out.println(matrix2);
        return matrix2;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new NeuralNetworkDriver(), args);
        System.exit(res);
    }

}
