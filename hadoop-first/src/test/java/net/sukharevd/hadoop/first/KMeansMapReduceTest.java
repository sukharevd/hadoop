package net.sukharevd.hadoop.first;

import static org.mockito.Mockito.*;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import net.sukharevd.hadoop.first.KMeansMapReduce.KMeansReducer;
import net.sukharevd.hadoop.first.KMeansMapReduce.KMeansMapper;
import net.sukharevd.hadoop.first.KMeansMapReduce.Point;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.junit.Test;

public class KMeansMapReduceTest {

	private KMeansMapper mapClass = new KMeansMapReduce.KMeansMapper();
	private KMeansReducer reduce = new KMeansMapReduce.KMeansReducer();

	@Test
	@SuppressWarnings("unchecked")
	public void map() throws IOException, SecurityException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
		Field centroidsField = KMeansMapper.class.getDeclaredField("centroids");
		centroidsField.setAccessible(true);
		List<Point> points = new ArrayList<Point>();
		points.add(Point.valueOf(1L, "0.0;0.0"));
		points.add(Point.valueOf(2L, "4.0;4.0"));
		centroidsField.set(mapClass, points);
		OutputCollector<LongWritable, Point> output = mock(OutputCollector.class);
		LongWritable key = new LongWritable(5L);
		Text value1 = new Text("3;3");
		mapClass.map(key, value1, output, mock(Reporter.class));
		verify(output).collect(new LongWritable(2L), Point.valueOf(5L, "3;3"));
	}
	
	@Test
	@SuppressWarnings("unchecked")
	public void map2() throws IOException, SecurityException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
		Field centroidField = KMeansMapper.class.getDeclaredField("centroids");
		centroidField.setAccessible(true);
		List<Point> points = new ArrayList<Point>();
		points.add(Point.valueOf(1L, "0.0;0.0"));
		points.add(Point.valueOf(2L, "4.0;4.0"));
		centroidField.set(mapClass, points);
		OutputCollector<LongWritable, Point> output = mock(OutputCollector.class);
		LongWritable key = new LongWritable(5L);
		Text value1 = new Text("1;1");
		mapClass.map(key, value1, output, mock(Reporter.class));
		verify(output).collect(new LongWritable(1L), Point.valueOf(5L, "1;1"));
	}
	
	@Test
	@SuppressWarnings("unchecked")
	public void reduceNotConverged() throws IOException, SecurityException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
	    Field centroidsField = KMeansReducer.class.getDeclaredField("centroids");
        centroidsField.setAccessible(true);
        List<Point> points = new ArrayList<Point>();
        points.add(Point.valueOf(1L, "0.0;0.0"));
        points.add(Point.valueOf(2L, "4.0;4.0"));
        centroidsField.set(reduce, points);
	    
		OutputCollector<LongWritable, Point> output = mock(OutputCollector.class);
		LongWritable key = new LongWritable(4L);
		Iterator<Point> values = mock(Iterator.class);
		when(values.hasNext()).thenReturn(true, true, true, false);
		when(values.next()).thenReturn(
		        Point.valueOf(1L, "1.0;1.0"),
		        Point.valueOf(1L, "2.0;2.0"),
		        Point.valueOf(1L, "6.0;6.0")
		);
		Reporter reporter = mock(Reporter.class);
        reduce.reduce(key, values, output, reporter);
		verify(output).collect(key, Point.valueOf(key.get(), "3.0;3.0"));
		verify(reporter).incrCounter(KMeansMapReduce.Counters.ALL, 1L);
		verify(reporter, never()).incrCounter(KMeansMapReduce.Counters.CONVERGED, 1L);
	}
	
	@Test
    @SuppressWarnings("unchecked")
    public void reduceConverged() throws IOException, SecurityException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
        Field centroidsField = KMeansReducer.class.getDeclaredField("centroids");
        centroidsField.setAccessible(true);
        List<Point> points = new ArrayList<Point>();
        points.add(Point.valueOf(1L, "0.0;0.0"));
        points.add(Point.valueOf(4L, "3.0;3.0"));
        centroidsField.set(reduce, points);
        
        OutputCollector<LongWritable, Point> output = mock(OutputCollector.class);
        LongWritable key = new LongWritable(4L);
        Iterator<Point> values = mock(Iterator.class);
        when(values.hasNext()).thenReturn(true, true, true, false);
        when(values.next()).thenReturn(
                Point.valueOf(1L, "1.0;1.0"),
                Point.valueOf(1L, "2.0;2.0"),
                Point.valueOf(1L, "6.0;6.0")
        );
        Reporter reporter = mock(Reporter.class);
        reduce.reduce(key, values, output, reporter);
        verify(output).collect(key, Point.valueOf(key.get(), "3.0;3.0"));
        verify(reporter).incrCounter(KMeansMapReduce.Counters.ALL, 1L);
        verify(reporter).incrCounter(KMeansMapReduce.Counters.CONVERGED, 1L);
    }
	
}
