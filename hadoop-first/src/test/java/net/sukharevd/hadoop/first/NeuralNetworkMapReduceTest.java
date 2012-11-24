package net.sukharevd.hadoop.first;

import static org.junit.Assert.assertArrayEquals;

import java.io.IOException;

import net.sukharevd.hadoop.entities.Matrix;
import net.sukharevd.hadoop.first.NeuralNetworkMapReduce.NeuralNetworkMapper;
import net.sukharevd.hadoop.first.NeuralNetworkMapReduce.NeuralNetworkReducer;

import org.junit.Test;

public class NeuralNetworkMapReduceTest {

	private NeuralNetworkMapper mapClass = new NeuralNetworkMapReduce.NeuralNetworkMapper();
	private NeuralNetworkReducer reduce = new NeuralNetworkMapReduce.NeuralNetworkReducer();

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
	}
	

}
