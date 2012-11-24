package net.sukharevd.hadoop.first;

import static net.sukharevd.hadoop.util.JamaExt.addConstant;
import static net.sukharevd.hadoop.util.JamaExt.addOneColumn;
import static net.sukharevd.hadoop.util.JamaExt.generateZ;
import static net.sukharevd.hadoop.util.JamaExt.log;
import static net.sukharevd.hadoop.util.JamaExt.removeFirstColumn;
import static net.sukharevd.hadoop.util.JamaExt.square;
import static net.sukharevd.hadoop.util.JamaExt.sum;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import net.sukharevd.hadoop.entities.Matrix;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public final class NeuralNetworkMapReduce extends Configured implements Tool {
    
    public final static double EPSILON = 1E-5;
    
    private List<Map<String, Long>> columnLabelsReplacements = new ArrayList<Map<String, Long>>();
    
    private static List<Jama.Matrix> readThetasFromThetasDir(JobConf job) {
        assert (job.get("thetas.path") != null);
        String filenames = "";
        try {
            FileSystem fs = FileSystem.get(job);
            Path centroidsPath = new Path(job.get("thetas.path"));
            if (fs.isFile(centroidsPath)) {
                filenames += centroidsPath.toString();
                return new ArrayList<Jama.Matrix>(readFile(job, fs, centroidsPath).values());
            } else { // isDirectory
                FileStatus[] listStatus = fs.listStatus(centroidsPath);
                List<Jama.Matrix> matrices = new ArrayList<Jama.Matrix>();
                for (FileStatus fileStatus : listStatus) {
                    if (fileStatus.getPath().toString().contains("part-")) {
                        filenames += fileStatus.getPath().toString();
                        SortedMap<Integer, Jama.Matrix> matrixMap = new TreeMap<Integer, Jama.Matrix>();
                        matrixMap.putAll(readFile(job, fs, fileStatus.getPath()));
                        matrices.addAll(matrixMap.values());
                    }
                }
                return matrices;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        throw new IllegalStateException("Read files: [" + filenames + "] from [" + job.get("thetas.path") + "]. See previous exceptions");
    }

    private static SortedMap<Integer, Jama.Matrix> readFile(JobConf job, FileSystem fs, Path thetasPath) throws IOException {
        SequenceFile.Reader thetasReader = new SequenceFile.Reader(fs, thetasPath, job);
        IntWritable layerId = new IntWritable();
        Text text = new Text();
        Matrix matrix = new Matrix();
        SortedMap<Integer, Jama.Matrix> matrices = new TreeMap<Integer, Jama.Matrix>();
        while (thetasReader.next(layerId, text)) {
            matrix = Matrix.valueOf(text.toString().split("\t")[0]);
            matrices.put(layerId.get(), new Jama.Matrix(matrix.getItems()));
        }
        assert (matrices.size() > 1);
        assert (!matrices.get(0).equals(matrices.get(1))); // assumption that there's at least one hidden layer
        thetasReader.close();
        return matrices;
    }

    public static class NeuralNetworkMapper extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
        private List<Jama.Matrix> thetas;
        private int k;
        private int hl;
        private double lambda;

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
            
            // J = -sum(sum(z .* log(a3) + (1-z) .* log(1.0 - a3))) ./ m + lambda * (sum(sumsq(Theta1(:,2:end)))+sum(sumsq(Theta2(:,2:end)))) / (2*m);
            double J = -sum(Y.arrayTimes(log(out)).plus(addConstant(Y.uminus(), 1).arrayTimes(log( addConstant(out.uminus(), 1)))));
            for (int i = 0; i < thetas.size(); i++) {
                J += lambda * sum(square(removeFirstColumn(thetas.get(0))));
            }
            
            Jama.Matrix[] delta = new Jama.Matrix[1 + hl]; // for each hidden unit and for output unit.
            delta[hl] = out.minus(Y); // last
            for (int i = hl - 1; i >= 0; i--) {
                delta[i] = delta[i+1].times(removeFirstColumn(thetas.get(i+1))).arrayTimes(sigmoidGradient(z[i+1]));
            }
            Jama.Matrix[] results = new Jama.Matrix[1 + hl];
            for (int i = 0; i <= hl; i++) {
                results[i] = delta[i].transpose().times(a[i]);
                output.collect(new IntWritable(i), new Text(new Matrix(results[i].getArray()) + "\t" + Double.toString(J) + "\t" + 1));
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

    public static class NeuralNetworkCombiner extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
        
        @Override
        public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter)
                throws IOException {
            Jama.Matrix result = null;
            long counter = 0;
            double J = 0d;
            while (values.hasNext()) {
                String value = values.next().toString();
                String[] split = value.split("\t");
                String matrixi = split[0];
                String Ji = split[1];
                String amounti = split[2];
                Matrix matrix = Matrix.valueOf(matrixi);
                J += Double.parseDouble(Ji);
                if (result == null) {
                    result = new Jama.Matrix(matrix.getItems());
                } else {
                    result.plusEquals(new Jama.Matrix(matrix.getItems()));
                }
                counter += Long.parseLong(amounti);
            }
            output.collect(key, new Text(new Matrix(result.getArray()).toString() + "\t" + J + "\t" + counter));
        }
    }

    
    public static class NeuralNetworkReducer extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
        private List<Jama.Matrix> thetas;
        private double alpha;
        private double lambda;

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
            long counter = 0;
            double J = 0d;
            while (values.hasNext()) {
                String value = values.next().toString();
                String[] split = value.split("\t");
                Matrix matrix = Matrix.valueOf(split[0]);
                J += Double.parseDouble(split[1]);
                if (result == null) {
                    result = new Jama.Matrix(matrix.getItems());
                } else {
                    result.plusEquals(new Jama.Matrix(matrix.getItems()));
                }
                counter += Long.parseLong(split[2]);
            }
            J /= counter;
            System.out.println(key.toString() + ": " + J + '\t' + counter + "\t" + Arrays.toString(result.getArray()[0]));
            Jama.Matrix delta = result.times(alpha/counter);
            Jama.Matrix regularization = thetas.get(key.get()).times(lambda/counter);
            thetas.get(key.get()).minusEquals(delta.plus(regularization));
            output.collect(key, new Text(new Matrix(thetas.get(key.get()).getArray()).toString() + "\t" + J + "\t" + counter));
        }
    }

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
        //Path initThetaPath = (args.length >= 10) ? new Path(args[9]): null;
        System.out.println("n = " + n + "\nhl = " + hl + "\nhu = " + u + "\nk = " + k + "\nalpha = " + alpha + "\nlambda = "
            + lambda + "\nmaxIteration = " + maxIteration);
        Configuration conf = getConf();
        JobConf job = new JobConf(conf, NeuralNetworkMapReduce.class);
        job.set("thetas.path", thetasPath.suffix("/it0").toString());
        //normalizeDataSet(conf, job, in);
        generateInitThetas(conf, job, n, hl, u, k);

        long i;
        for (i = 0; i < maxIteration; i++) {
            conf = getConf();
            job = new JobConf(conf, NeuralNetworkMapReduce.class);
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
    
    /*    private void generateInitThetas(Configuration conf, JobConf job, Path in, int m, int h, int u, int k) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    Path thetasPath = new Path(job.get("thetas.path"));
    SequenceFile.Writer thetasWriter = SequenceFile.createWriter(fs, conf, thetasPath, IntWritable.class, Matrix.class);
    thetasWriter.append(new IntWritable(0), Matrix.valueOf("0.8174629387699757 0.2435950611101989 0.22073891599227313 0.9526024622142223;0.37734201725821326 0.843365338762703 0.3984911596576973 0.5989882499905892;0.7030912654502783 0.35063263224308583 0.26458675296606915 0.008666681317597846;0.1397208621182332 0.8419598179297162 0.020085140145313596 0.2613557183438735"));
    thetasWriter.append(new IntWritable(1), Matrix.valueOf("0.13946395085521646 0.7661558078982684 0.32178095012486574 0.06290819344487453 0.4426928460044566;0.5235017778011603 0.857436506092054 0.1772015636851122 0.001424240427704171 0.04465404966870612;0.05943136807714278 0.7399826094462248 0.3132907078651943 0.2619543319729256 0.08417915309671398"));
    thetasWriter.close();
    }*/

/*    private Matrix generateThetasI2(int outputs, int inputs, int generalOutputs, int generalInputs) {
        double e = Math.sqrt(6d) / Math.sqrt(inputs + outputs);
        double[][] matrix = new double[outputs][inputs];
        for (int i = 0; i < outputs; i++) {
            for (int j = 0; j < inputs; j++) {
                matrix[i][j] = (new Random().nextInt(Math.abs(outputs - inputs) + 1) + outputs) * 2d * e - e;
            }
        }
        Matrix matrix2 = new Matrix(matrix);
        System.out.println(matrix2);
        return matrix2;
    }*/

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


    private void normalizeDataSet(Configuration conf, JobConf job, Path in) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream inputStream = fs.open(in);
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        Path numerizedInputPath = in.suffix(".numerized");
        FSDataOutputStream out = fs.create(numerizedInputPath);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out));
        
        System.out.println("Eliminate nominal values");
        long counter = 0;
        Jama.Matrix means = null;
        while (true) {
                String line = br.readLine();
                if (line == null) break;
                line = line.substring(0, line.length()-1); // remove the point
                String[] split = line.split(",");
                eliminateNominalValues(split);
                StringBuilder builder = new StringBuilder(split[0]);
                for (int i = 1; i < split.length; i++) {
                    builder.append('\t').append(split[i]);
                }
                builder.append('\n');
                bw.write(builder.toString());
        }
        br.close();
        bw.flush();
        bw.close();
        
        Path InputFeaturesMapPath = in.suffix(".featuresMap");
        out = fs.create(InputFeaturesMapPath);
        out.writeUTF(columnLabelsReplacements.toString());
        out.close();
        
        System.out.println("Means");
        inputStream = fs.open(numerizedInputPath);
        br = new BufferedReader(new InputStreamReader(inputStream));
        while (true) {
                String line = br.readLine();
                if (line == null) break;
                String[] split = line.split("\t");
                if (means == null) {
                    means = new Jama.Matrix(1, split.length);
                }
                Jama.Matrix vector = convertToVector(split);
                System.out.println(counter + ": " +means.getColumnDimension() + "x" + means.getRowDimension());
                System.out.println(counter + ": " +vector.getColumnDimension() + "x" + vector.getRowDimension());
                means = means.plus(vector);
                counter++;
        }
        means.timesEquals(1d/counter);
        br.close();
        
        System.out.println("Sigmas");
        Jama.Matrix sigmas = new Jama.Matrix(1, means.getColumnDimension());
        inputStream = fs.open(numerizedInputPath);
        br = new BufferedReader(new InputStreamReader(inputStream));
        while (true) {
                String line = br.readLine();
                if (line == null) break;
                String[] split = line.split("\t");
                Jama.Matrix vector = convertToVector(split);
                sigmas.plusEquals(vector.minus(means).arrayTimes(vector.minus(means)));
        }
        sigmas.timesEquals(1d/counter);
        for (int i = 0; i < sigmas.getArray()[0].length; i++) {
            if (sigmas.get(0, i) == 0) {
                sigmas.set(0, i, 1.0);
                System.out.println("WARN: sigma of feature " + i + " is zero, mean is " + means.get(0,i));
            }
        }
        
        for (int i = 0; i < sigmas.getColumnDimension(); i++) {
            sigmas.set(0, i, Math.sqrt(sigmas.get(0, i)));
        }
        br.close();
        for (int i = 0; i < means.getArray().length; i++) {
            System.out.println("Means: " + Arrays.toString(means.getArray()[i]));
        }
        for (int i = 0; i < means.getArray().length; i++) {
            System.out.println("Sigmas: " + Arrays.toString(sigmas.getArray()[i]));
        }
        
        System.out.println("Normalized");
        inputStream = fs.open(numerizedInputPath);
        br = new BufferedReader(new InputStreamReader(inputStream));
        Path normalizedInputPath = in.suffix(".normalized");
        out = fs.create(normalizedInputPath);
        bw = new BufferedWriter(new OutputStreamWriter(out));
        while (true) {
            String line = br.readLine();
            if (line == null) break;
            String[] split = line.split("\t");
            Jama.Matrix vector = convertToVector(split);
            vector.minusEquals(means).arrayRightDivideEquals(sigmas);
            StringBuilder builder = new StringBuilder(Double.toString(vector.get(0, 0)));
            for (int i = 1; i < split.length; i++) {
                builder.append('\t').append(vector.get(0, i));
            }
            bw.write(builder.toString());
            bw.newLine();
        }
        br.close();
        bw.flush();
        bw.close();
    }

    private Jama.Matrix convertToVector(String[] split) {
        double[] vectorArray = new double[split.length];
        for (int i = 0; i < split.length; i++) {
            vectorArray[i] = Double.valueOf(split[i]);
        }
        return new Jama.Matrix(new double[][] { vectorArray });
    }

    private void eliminateNominalValues(String[] split) {
        if (columnLabelsReplacements.isEmpty()) {
            for (int i = 0; i < split.length; i++) {
                columnLabelsReplacements.add(new HashMap<String, Long>());
            }
        }
        for (int i = 0; i < split.length; i++) {
            if (!split[i].matches("[0-9\\.]+")) {
                if (columnLabelsReplacements.get(i).get(split[i]) != null) {
                    split[i] = Long.valueOf(columnLabelsReplacements.get(i).get(split[i])).toString();
                } else {
                    long last = 0;
                    for (Long value: columnLabelsReplacements.get(i).values()) {
                        if (last <= value) {
                            last = value + 1;
                        }
                    }
                    columnLabelsReplacements.get(i).put(split[i], last);
                    split[i] = Long.valueOf(last).toString();
                }
            }
        }
    }
    
    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new NeuralNetworkMapReduce(), args);
        System.exit(res);
    }

}
