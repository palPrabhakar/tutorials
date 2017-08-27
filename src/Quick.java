import edu.princeton.cs.algs4.*;

public class Quick {
	
	public static void sort(int[] a) {
		StdRandom.shuffle(a);
		int n = a.length;
		if (n == 0) return;
		sort(a, 0, n - 1);
	}
	
	public static void sort(int[] a, int lo, int hi) {
		if (hi <= lo) return; 
		int k = lo;
		int i = lo + 1;
		int j = hi;
		
		while(j >= i) {
			while(i <= hi && less(a[i],a[k])) i++;
			while(j > lo && a[k] <= a[j]) j--;
			if(i < j) exch(a, i, j);
		}
		exch(a, k, j);
		
		/*
		for (int x : a) {
			StdOut.printf("%d ", x);
		}
		StdOut.printf("Fuckers : %d \n" , a[j]);
		*/
		sort(a, lo, j-1);
		sort(a, j+1, hi);
		
		
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
		int n = Integer.parseInt(args[0]);
		int[] arr = new int[n];
		for (int i = 0; i < n; i++){
			arr[i] = i;
		}
		
		StopwatchCPU timer = new StopwatchCPU();
		sort(arr);
		double time = timer.elapsedTime();
		
		System.out.printf("%b : %f \n",isSorted(arr),time);
		/*
		for (int x : arr) {
			StdOut.println(x);
		}
		*/
	}

}

