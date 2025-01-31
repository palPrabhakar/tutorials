import java.util.Iterator;
import edu.princeton.cs.algs4.*;

public class ResizingArrayDeque<Item> implements Iterable<Item>{
	
	private Item[] a = (Item[]) new Object[1];
	private int N; 
	
	public boolean isEmpty() {
		return N == 0;
	}
	
	public int size(){
		return N;
	}
	
	private void resize(int max){
		Item[] temp = (Item[]) new Object[max];
		for(int i = 0; i < N; i++){
			temp[i] = a[i];
		}
		a = temp;
	}
	
	public void pushleft(Item item) {
		if(N == a.length) resize(2*a.length);
		for(int i = N-1; i >= 0; i--){
			a[i+1] = a[i];
		}
		a[0] = item;
		N++;
	}
	
	public void pushright(Item item) {
		if(N == a.length) resize(2*a.length);
		a[N++] = item;
	}
	
	public Item popleft() {
		Item item = a[0];
		a[0] = null;
		for(int i = 0; i < N-1; i++){
			a[i] = a[i+1];
		}
		if(N>0 && N <= a.length/4) resize(a.length/2);
		N--;
		return item;
	}
	
	public Item popright() {
		Item item = a[--N];
		a[N] = null;
		if(N > 0 && N <= a.length/4) resize(a.length/2);
		return item;
	}
	
	public Iterator<Item> iterator(){
		return new dequeIterator();
	}
	
	public class dequeIterator implements Iterator<Item> {
		private int i = N;
		public boolean hasNext() { return i > 0; }
		public void remove() { }
		public Item next() { return a[--i]; }
	}
	
	public static void main(String[] args) {
		ResizingArrayDeque<Integer> x = new ResizingArrayDeque<Integer>();
		
		x.pushleft(1);
		//StdOut.println(x.popleft());
		x.pushleft(2);
		x.pushleft(3);
		x.pushleft(4);
		x.pushleft(5);
		x.pushright(6);
		x.pushright(7);
		x.pushright(8);
		x.pushright(9);
		x.pushright(10);
		
		
		/*
		StdOut.println(x.popleft());
		//StdOut.println(x.popright());
		StdOut.println(x.popleft());
		StdOut.println(x.popleft());
		StdOut.println(x.popleft());
		StdOut.println(x.popleft());
		StdOut.println(x.popright());
		StdOut.println(x.popright());
		StdOut.println(x.popright());
		StdOut.println(x.popright());
		StdOut.println(x.popright());
		*/
		
		for(Integer y : x ) StdOut.println(y);
	}
}
