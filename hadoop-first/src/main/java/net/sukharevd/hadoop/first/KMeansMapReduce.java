package net.sukharevd.hadoop.first;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public final class KMeansMapReduce extends Configured implements Tool {
    
    /** The minimal shift of centroid for continue iteration, if the shift is less then epsilon the algorithm converged. */
    public final static double EPSILON = 1E-5;
    
    static enum Counters {
        ALL, CONVERGED
    }

    public static class Point implements WritableComparable<Point> {
        private Long id;
        private List<Double> coordinates = new ArrayList<Double>();

        public Point() {
        }

        public Point(Point clone) {
            id = clone.id;
            coordinates = new ArrayList<Double>(clone.coordinates);
        }

        public Long getId() {
            return id;
        }

        public void setId(Long id) {
            this.id = id;
        }

        public List<Double> getCoordinates() {
            return coordinates;
        }

        public void setCoordinates(List<Double> coordinates) {
            this.coordinates = coordinates;
        }

        public static Point valueOf(Long id, String coordinatesStr) {
            Point point = new Point();
            point.setId(id);
            // System.out.println("Centroid::valueOf(...) obtained " + coordinatesStr);
            String[] split = coordinatesStr.split(";");
            for (String string : split) {
                point.coordinates.add(Double.valueOf(string));
            }
            return point;
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            id = in.readLong();
            String coordinatesStr = in.readUTF();
            // System.out.println("Centroid::readFields(...) read " + coordinatesStr);
            String[] split = coordinatesStr.split(";");
            coordinates.clear();
            for (String string : split) {
                coordinates.add(Double.valueOf(string));
            }
        }

        @Override
        public void write(DataOutput out) throws IOException {
            StringBuilder coordinatesStr = new StringBuilder();
            if (!coordinates.isEmpty()) {
                coordinatesStr.append(coordinates.get(0));
            }
            for (int i = 1; i < coordinates.size(); i++) {
                coordinatesStr.append(';').append(coordinates.get(i));
            }
            out.writeLong(id);
            out.writeUTF(coordinatesStr.toString());
        }

        @Override
        public int compareTo(Point o) {
            return id.compareTo(o.id);
        }

        public void plus(Point point) {
            assert (coordinates.size() == point.coordinates.size());
            for (int i = 0; i < coordinates.size(); i++) {
                coordinates.set(i, coordinates.get(i) + point.coordinates.get(i));
            }
        }

        public void div(long value) {
            for (int i = 0; i < coordinates.size(); i++) {
                coordinates.set(i, coordinates.get(i) / value);
            }
        }

        public static void plus(Point point1, Point point2) {
            assert (point1.coordinates.size() == point2.coordinates.size());
            Point point = new Point();
            point.setId(point1.id);
            for (int i = 0; i < point1.coordinates.size(); i++) {
                point.coordinates.add(i, point1.coordinates.get(i) + point2.coordinates.get(i));
            }
        }

        public static void div(Point point, long value) {
            Point res = new Point();
            res.setId(point.id);
            for (int i = 0; i < point.coordinates.size(); i++) {
                res.coordinates.add(i, point.coordinates.get(i) / value);
            }
        }

        @Override
        public String toString() {
            assert (!coordinates.isEmpty());
            StringBuilder builder = new StringBuilder();
            builder.append(id).append('\t');
            builder.append(coordinates.get(0)); // there should be at least one coordinate
            for (int i = 1; i < coordinates.size(); i++) {
                builder.append('\t').append(coordinates.get(i));
            }
            return builder.toString();
        }

        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof Point)) {
                return false;
            }
            Point rhs = (Point) obj;
            return id.equals(rhs.id) && coordinates.equals(rhs.coordinates);
        }

        @Override
        public int hashCode() {
            double sum = id;
            for (Double c : coordinates) {
                sum += c;
            }
            return Double.valueOf(sum).hashCode();
        }
    }

    private static List<Point> readCentroidsFromCentroidsDir(JobConf job) {
        assert (job.get("centroids.path") != null);
        List<Point> centroids = new ArrayList<Point>();
        String filenames = "";
        try {
            FileSystem fs = FileSystem.get(job);
            Path centroidsPath = new Path(job.get("centroids.path"));
            if (fs.isFile(centroidsPath)) {
                filenames += centroidsPath.toString();
                centroids.addAll(readFile(job, fs, centroidsPath));
            } else { // isDirectory
                FileStatus[] listStatus = fs.listStatus(centroidsPath);
                for (FileStatus fileStatus : listStatus) {
                    if (fileStatus.getPath().toString().contains("part-")) {
                        filenames += fileStatus.getPath().toString();
                        centroids.addAll(readFile(job, fs, fileStatus.getPath()));
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (centroids.isEmpty())
            throw new IllegalStateException("Read files: " + filenames + " from " + job.get("centroids.path") + ". Got nothing");
        return centroids;
    }

    private static List<Point> readFile(JobConf job, FileSystem fs, Path centroidsPath) throws IOException {
        List<Point> centroids = new ArrayList<Point>();
        SequenceFile.Reader centroidsReader = new SequenceFile.Reader(fs, centroidsPath, job);
        LongWritable nullWritable = new LongWritable();
        Point centroid = new Point();
        while (centroidsReader.next(nullWritable, centroid)) {
            centroids.add(new Point(centroid));
            // System.out.println("Mapper#conf():: Added centroid:" + centroid.toString());
        }
        assert (centroids.size() > 1);
        assert (!centroids.get(0).equals(centroids.get(1)));
        centroidsReader.close();
        return centroids;
    }
    
    private static double distanceBetween(Point centroid, Point point) {
        double d = 0d;
        for (int i = 0; i < centroid.getCoordinates().size(); i++) {
            d += Math.pow(centroid.getCoordinates().get(i) - point.getCoordinates().get(i), 2);
        }
        return d;
    }
    
    public static class KMeansMapper extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, Point> {
        private List<Point> centroids;
        private LongWritable nearestCentroidId = new LongWritable();

        @Override
        public void configure(JobConf job) {
            super.configure(job);
            centroids = readCentroidsFromCentroidsDir(job);
        }

        @Override
        public void map(LongWritable key, Text value, OutputCollector<LongWritable, Point> output, Reporter reporter) throws IOException {
            assert (!centroids.isEmpty());
            Point point = Point.valueOf(key.get(), value.toString());
            double minDistance = Double.MAX_VALUE;
            Point nearestCentroid = centroids.get(0);
            for (Point centroid : centroids) {
                double distanceBetween = distanceBetween(centroid, point);
                if (minDistance > distanceBetween) {
                    minDistance = distanceBetween;
                    nearestCentroid = centroid;
                }
            }
            nearestCentroidId.set(nearestCentroid.getId());
            output.collect(nearestCentroidId, point);
        }

    }
    
    public static class KMeansReducer extends MapReduceBase implements Reducer<LongWritable, Point, LongWritable, Point> {
        private List<Point> centroids;

        @Override
        public void configure(JobConf job) {
            super.configure(job);
            centroids = readCentroidsFromCentroidsDir(job);
        }

        @Override
        public void reduce(LongWritable key, Iterator<Point> values, OutputCollector<LongWritable, Point> output, Reporter reporter)
                throws IOException {
            Point newCentroid = new Point();
            newCentroid.setId(key.get());
            long count = 0;
            while (values.hasNext()) {
                Point point = values.next();
                if (newCentroid.getCoordinates().isEmpty()) {
                    for (int i = 0; i < point.getCoordinates().size(); i++) {
                        newCentroid.getCoordinates().add(0d);
                    }
                }
                count++;
                newCentroid.plus(point);
            }
            newCentroid.div(count);
            output.collect(new LongWritable(newCentroid.getId()), newCentroid);
            reporter.incrCounter(Counters.ALL, 1L);
            if (converged(newCentroid)) {
                reporter.incrCounter(Counters.CONVERGED, 1L);
            }
        }

        private boolean converged(Point newCentroid) {
            for (Point oldCentroid : centroids) {
                boolean sameCentroid = oldCentroid.getId().equals(newCentroid.getId());
                if (sameCentroid) {
                    return distanceBetween(oldCentroid, newCentroid) < EPSILON;
                }
            }
            return false;
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        JobConf job = new JobConf(conf, KMeansMapReduce.class);
        Path in = new Path(args[0]);
        Path center = new Path(args[2]);
        long k = Long.valueOf(args[3]);
        long maxIteration = 20;
        job.set("centroids.path", center.suffix("/it0").toString());
        generateInitCentroids(conf, job, in, k);

        for (long i = 0; i < maxIteration; i++) {
            conf = getConf();
            job = new JobConf(conf, KMeansMapReduce.class);
            job.setLong("clustering.iteration", Long.valueOf(i));
            Path prevout = center.suffix("/it" + job.get("clustering.iteration"));
            Path out = center.suffix("/it" + (i + 1));
            job.set("centroids.path", prevout.toString());
            FileInputFormat.setInputPaths(job, in);
            FileOutputFormat.setOutputPath(job, out);
            job.setJobName("KMeans MapReduce (iteration" + i + ")");
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setInputFormat(TextInputFormat.class);
            job.setOutputFormat(SequenceFileOutputFormat.class);
            job.setOutputKeyClass(LongWritable.class);
            job.setOutputValueClass(Point.class);
            RunningJob runningJob = JobClient.runJob(job);
            long convergedCounter = runningJob.getCounters().getCounter(Counters.CONVERGED);
            long allCounter = runningJob.getCounters().getCounter(Counters.ALL);
            if (convergedCounter == allCounter) {
                System.out.println("Converged at " + i + "th iteration.");
                break;
            } else {
                System.out.println("Converged " + convergedCounter + " of " + allCounter);
            }
        }

        conf = getConf();
        job = new JobConf(conf, KMeansMapReduce.class);
        job.setLong("clustering.iteration", Long.valueOf(maxIteration));
        Path prevout = center.suffix("/it" + job.get("clustering.iteration"));
        job.set("centroids.path", prevout.toString());
        Path out = new Path(args[1]);
        FileInputFormat.setInputPaths(job, in);
        FileOutputFormat.setOutputPath(job, out);
        job.setJobName("KMeans MapReduce (final step)");
        job.setMapperClass(KMeansMapper.class);
        job.setInputFormat(TextInputFormat.class);
        job.setOutputFormat(TextOutputFormat.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Point.class);
        job.setMaxReduceAttempts(0);
        JobClient.runJob(job);

        return 0;
    }

    private void generateInitCentroids(Configuration conf, JobConf job, Path in, long k) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream inputStream = fs.open(in);
        BufferedReader bin = new BufferedReader(new InputStreamReader(inputStream));
        Path centroidsPath = new Path(job.get("centroids.path"));
        SequenceFile.Writer centroidsWriter = SequenceFile.createWriter(fs, conf, centroidsPath, LongWritable.class, Point.class);
        for (long i = 1; i <= k; i++) {
            // there're should be at least k objects in data set
            Point centroid = Point.valueOf(i, bin.readLine());
            System.out.println("Generated centroid: " + centroid);
            centroidsWriter.append(new LongWritable(i), centroid);
        }
        inputStream.close();
        centroidsWriter.close();
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new KMeansMapReduce(), args);
        System.exit(res);
    }

}
