import edu.princeton.cs.algs4.*;

public class MaxPQ {
	private int[] pq;
	private int n = 1;
	
	public MaxPQ(int capacity) {
		pq = new int[capacity+1];
	}
	
	public int removeMax(){
		int val = pq[1];
		exch(1, --n);
		pq[n] = Integer.MIN_VALUE;
		sink(1);
		return val;
	}
	
	public void insert(int val) {
		pq[n] = val;
		swim(n);
		n++;
	}
	
	public void swim(int k) {
		if (k > 1 && less(k/2, k)) {
			exch(k/2, k);
			swim(k/2);
		}
	}
	
	public void sink(int k) {
		if(2*k > n) return;
		int m;
		if (less(2*k+1, 2*k)) m = 2*k;
		else m = 2*k+1;
		
		if (less(k, m)){
			exch(m, k);
			sink(m);
		}
 	}
	
	public boolean isEmpty() {
		return n-1 == 0;
	}
	
	public int size() {
		return n-1;
	}
	
	public boolean less(int i, int j) {
		return pq[i] < pq[j];
	}
	
	public void exch(int i, int j) {
		int temp = pq[i];
		pq[i] = pq[j];
		pq[j] = temp;
	}
	
	public void print() {
		for(int x = 1; x < n; x++) {
			StdOut.printf("%d ",pq[x]);
		}
		StdOut.printf("\n");
	}
	
	public static void main(String[] args) {
		MaxPQ pq = new MaxPQ(Integer.parseInt(args[0]));
		int i = 0;
		while (i < Integer.parseInt(args[0])) {
			int val = StdIn.readInt();
			pq.insert(val);
			//pq.print();
			i++;
		}
		pq.print();
		
		i = 0;
		while(i < Integer.parseInt(args[0])) {
			StdOut.printf("%d \n",pq.removeMax());
			//pq.print();
			i++;
		}
		
	}
	
}
