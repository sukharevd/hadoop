package net.sukharevd.hadoop.first;

import static org.mockito.Mockito.*;

import java.io.IOException;
import java.util.Iterator;

import net.sukharevd.hadoop.first.ThirdMapReduce.MapClass;
import net.sukharevd.hadoop.first.ThirdMapReduce.Reduce;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.junit.Test;

public class ThirdMapReduceTest {

	private MapClass mapClass = new ThirdMapReduce.MapClass();
	private Reduce reduce = new ThirdMapReduce.Reduce();

	@Test
	@SuppressWarnings("unchecked")
	public void map() throws IOException {
		OutputCollector<IntWritable, IntWritable> output = mock(OutputCollector.class);
		Text key = mock(Text.class);
		Text value1 = new Text("3");
		IntWritable value2 = new IntWritable(3);
		IntWritable one = new IntWritable(1);
		mapClass.map(key, value1, output, mock(Reporter.class));
		verify(output).collect(value2, one);
	}
	
	@Test
	@SuppressWarnings("unchecked")
	public void reduce() throws IOException {
		OutputCollector<IntWritable, IntWritable> output = mock(OutputCollector.class);
		IntWritable key = mock(IntWritable.class);
		Iterator<IntWritable> values = mock(Iterator.class);
		when(values.hasNext()).thenReturn(true, true, true, false);
		when(values.next()).thenReturn(new IntWritable(1), new IntWritable(2), new IntWritable(3));
		reduce.reduce(key, values, output, mock(Reporter.class));
		verify(output).collect(key, new IntWritable(1+2+3));
	}
	
}
