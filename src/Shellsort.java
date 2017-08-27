import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StopwatchCPU;

public class Shellsort {
	
	public static void sort(int[] a) {
		int n = a.length;
		int h = 1;
		while (h < n/3) h = 3*h + 1;
		
		while (h > 0) {
			for (int i = h; i < n; i++) {
				for (int j = i; j >= h && less(a[j], a[j-h]); j -= h) {
					exch(a, j, j-h);
				}
			}
			//StdOut.println(h);
			h = h/3;
		}
	}
	
	public static void exch(int[] x, int i, int j) {
		int temp =  x[i];
		x[i] = x[j];
		x[j] = temp;
	}
	
	public static boolean less(int a, int b) {
		return a < b;
	}
	
	public static boolean isSorted(int[] a) {
		int n = a.length;
		
		for (int i = 0; i < n-1; i++)
			if (less(a[i+1], a[i])) return false;
		
		return true;
	}
	
	public static void main(String[] args) {
		int N = Integer.parseInt(args[0]);
		int[] arr = new int[N];
		
		for (int i = 0; i < N; i++) {
			arr[i] = StdRandom.uniform(N);
		}
		
		/*
		for(int x : arr) {
			StdOut.println(x);
		}
		*/
		
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

