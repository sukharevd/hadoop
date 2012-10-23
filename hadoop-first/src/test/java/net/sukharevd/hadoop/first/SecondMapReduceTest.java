package net.sukharevd.hadoop.first;

import static org.mockito.Mockito.*;

import java.io.IOException;
import java.util.Iterator;

import net.sukharevd.hadoop.first.SecondMapReduce.MapClass;
import net.sukharevd.hadoop.first.SecondMapReduce.Reduce;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.junit.Test;

public class SecondMapReduceTest {

	private MapClass mapClass = new SecondMapReduce.MapClass();
	private Reduce reduce = new SecondMapReduce.Reduce();

	@Test
	@SuppressWarnings("unchecked")
	public void map() throws IOException {
		OutputCollector<Text, Text> output = mock(OutputCollector.class);
		Text key = mock(Text.class);
		Text value = mock(Text.class);
		mapClass.map(key, value, output, mock(Reporter.class));
		verify(output).collect(value, key);
	}
	
	@Test
	@SuppressWarnings("unchecked")
	public void reduce() throws IOException {
		OutputCollector<Text, IntWritable> output = mock(OutputCollector.class);
		Text key = mock(Text.class);
		Iterator<Text> values = mock(Iterator.class);
		when(values.hasNext()).thenReturn(true, true, true, false);
		when(values.next()).thenReturn(new Text("one"), new Text("two"), new Text("three"));
		reduce.reduce(key, values, output, mock(Reporter.class));
		verify(output).collect(key, new IntWritable(3));
	}
	
}
