import edu.princeton.cs.algs4.*;

public class Insertion {
	
	public static void sort(int[] a) {
		int n = a.length;
		
		
		for(int i = 1; i < n; i++) {
			for(int j = i; j > 0 && less(a[j], a[j-1]); j--) {
				exch(a, j, j-1);
			}
		}
		
	}
	
	public static void exch(int[] x, int i, int j) {
		int temp =  x[i];
		x[i] = x[j];
		x[j] = temp;
	}
	
	public static boolean less(int a, int b) {
		if (a < b) return true;
		else return false;
	}
	
	public static boolean isSorted(int[] a) {
		int n = a.length;
		
		for(int i = 0; i < n-1; i++)
			if(less(a[i+1], a[i])) return false;
		
		return true;
	}
	
	public static void main(String[] args){
		int N = Integer.parseInt(args[0]);
		int[] arr = new int[N];
		
		for(int i = 0; i < N; i++){
			arr[i] = StdRandom.uniform(10000);
		}
		
		Insertion.sort(arr);
		
		StdOut.println(isSorted(arr));
		
		/*
		for(int x : arr) {
			StdOut.println(x);
		}
		*/
	}
	
}
