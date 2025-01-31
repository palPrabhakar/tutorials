import edu.princeton.cs.algs4.*;

public class Merge {
	
	public static void merge (int[] a, int lo, int mid, int hi) {
		int[] aux = new int[hi - lo + 1];
		int i = lo;
		int j = mid;
	
		for(int k = 0; k <= hi -lo; k++) {
			if (i == mid) { aux[k] = a[j]; j++; }
			else if (j > hi) { aux[k] = a[i]; i++; }
			else if (less(a[i], a[j])) { aux[k] = a[i]; i++; }
			else { aux[k] = a[j]; j++; }
		}
		
		for(int k = lo; k <= hi; k++) {
			a[k] = aux[k-lo];
		}
			
	}
	
	public static void sort(int[] a) {
		int n = a.length;
		if (n == 0) return;
		sort(a, 0, n-1);
	}
	
	public static void sort(int[] a, int lo, int hi) {
		if (hi <= lo) return;
		int mid = lo + (hi - lo)/2;
		sort(a, lo, mid);
		sort(a, mid+1, hi);
		merge(a, lo, mid+1, hi);
	}
	
	public static boolean less(int a, int b) {
		if (a < b) return true;
		else return false;
	}
	
	public static boolean isSorted(int[] a) {
		int n = a.length;
		
		for (int i = 0; i < n-1; i++)
			if(less(a[i+1], a[i])) return false;
		
		return true;
	}

	public static void main(String[]  args) {
		int n = Integer.parseInt(args[0]);
		int[] arr = new int[n];
		
		for (int i = 0; i < n; i++) {
			arr[i] = StdRandom.uniform(n);
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
