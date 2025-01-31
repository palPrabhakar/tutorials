import edu.princeton.cs.algs4.*;

public class Heap {
	
	public static void sort(int[] a) {
		int n = a.length;
		
		for (int i = n/2; i >= 1; i--) {
			sink(a, i, n);
		}
		
		while (n > 1) {
			exch(a, 1, --n);
			sink(a, 1, n);
		}
		
	}
	
	public static void sink(int[] a, int k, int n) {
		if(2*k  >= n) return;
		
		int m = 2*k;
		if (m+1 < n && less(a[m], a[m+1])) m++;
		if (less(a[k], a[m])) exch(a, m, k);
		sink(a, m, n);

	}
	
	public static void exch(int[] pq, int i, int j) {
		int temp = pq[i];
		pq[i] = pq[j];
		pq[j] = temp;
	}
	
	public static boolean isSorted(int[] a) {
		int n = a.length;
		
		for(int i = 1; i < n-1; i++)
			if(less(a[i+1], a[i])) return false;
		
		return true;
	}
	
	public static boolean less(int a, int b) {
		return a < b;
	}
	
	public static void main(String[] args){
		
		//int[] arr = { 0, 256, 2, 5, 6, 7, 4, 8, 9, 56};
		
		int[] arr = new int[Integer.parseInt(args[0])];
		
		for(int i = 1; i < Integer.parseInt(args[0]); i++) {
			arr[i] = StdRandom.uniform(Integer.parseInt(args[0]));
		}
		
		StopwatchCPU timer = new StopwatchCPU();
		sort(arr);
		double time = timer.elapsedTime();
		
		System.out.printf("%b : %f \n",isSorted(arr),time);
		/*
		for(int x : arr) {
			StdOut.println(x);
		}
		*/
		
	}
}
