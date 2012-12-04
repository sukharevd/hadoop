package net.sukharevd.hadoop.first;

import static org.junit.Assert.assertArrayEquals;

import java.io.IOException;

import net.sukharevd.hadoop.entities.Matrix;
import net.sukharevd.hadoop.nn.NeuralNetworkDriver;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.util.ToolRunner;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkMapReduceTest {

	@Test
	public void valueOfMatrix() {
	    assertArrayEquals(new double[][]{{1,2,3},{4,5,6}}, Matrix.valueOf("1 2 3; 4 5 6").getItems());
	}

	@Test
	public void map() throws IOException, SecurityException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
//		Field centroidsField = KMeansMapper.class.getDeclaredField("centroids");
//		centroidsField.setAccessible(true);
//		List<Point> points = new ArrayList<Point>();
//		points.add(Point.valueOf(1L, "0.0;0.0"));
//		points.add(Point.valueOf(2L, "4.0;4.0"));
//		centroidsField.set(mapClass, points);
//		OutputCollector<LongWritable, Point> output = mock(OutputCollector.class);
//		LongWritable key = new LongWritable(5L);
//		Text value1 = new Text("3;3");
//		mapClass.map(key, value1, output, mock(Reporter.class));
//		verify(output).collect(new LongWritable(2L), Point.valueOf(5L, "3;3"));

//	    Jama.Matrix a = JamaExt.generateZ(4, 10);
//	    for (int j = 0; j < a.getArray().length; j++) {
//	        System.out.println(Arrays.toString(a.getArray()[j]));
//        }
	}
	
	@Before
	@After
	public void prepare() throws IOException {
	    Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        org.apache.hadoop.fs.Path file = new org.apache.hadoop.fs.Path("src/test/resources/outputs/it1/");
        fs.delete(file, true);
        file = new org.apache.hadoop.fs.Path("src/test/resources/outputs/it2/");
        fs.delete(file, true);
        fs.close();
	}
	
	@Test
	public void complexTest() throws Exception {
	    String[] args = new String[] {
	        "src/test/resources/input/kddcup.data.corrected.normalized.s",     // input data-set path
	        "src/test/resources/outputs/",                                     // output thetas directory
	        "1",                                                               // number of features
	        "1",                                                               // number of hidden layers
	        "2",                                                               // number of units in hidden layers
	        "4",                                                               // number of output units
	        "2",                                                               // alpha
	        "0.0",                                                             // lambda
	        "2"                                                                // number of iterations
	    };
	    ToolRunner.run(new Configuration(), new NeuralNetworkDriver(), args);
	    
	}

}
