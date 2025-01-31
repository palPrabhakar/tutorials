import	edu.princeton.cs.algs4.*;
import java.io.*;
import java.util.*;

public class Solution {
    
    public static int[] lex(String str){
        for(int j = str.length() - 2; j >= 0; j--) {
                for(int k = str.length() - 1; k > j; k--) {
                    if(str.charAt(j) < str.charAt(k)) {
                        //System.out.println("fuckYeah");
                        int[] arr = {j, k};
                        return arr;
                    }
                }
            }
        int[] arr = {-1, -1};
        return arr; 
    }
    
    public static void result(String str, int pos, int next, String res){
    	//System.out.println(pos);
    	//System.out.println(next);
        String result = "";
        result += str.subSequence(0, pos);
        char[] strc = new char[str.length()-pos];
        //char[] strcc = new char[strc.length-1];
        
        for(int i = 0; i < strc.length; i++){
        	if(pos + i ==next) strc[i] = ' ';
        	else
        		strc[i] = str.charAt(pos + i);
        }
        result += str.charAt(next);
        
        
        /*for(int i = 0; i < strcc.length; i++){
            if(i+pos == next){
            	strcc[i] = strc[i];
            }
            strcc[i] = strc[i];
        }*/
        Arrays.sort(strc);
        
        for(int i = 1; i <strc.length; i++){
            result += strc[i];
        }
        System.out.println(result.compareTo(res));
        //System.out.println(result);
    }

    public static void main(String[] args) {
        /* Enter your code here. Read input from STDIN. Print output to STDOUT. Your class should be named Solution.
        */
        In in = new In("input02.txt");
        In out = new In("output02.txt");
        int n = Integer.parseInt(in.readLine());
        for(int i = 0; i < n; i++) {
           // String str = "sqbnyzclwqawbpdtzincfawrwlfmefhatqeedicyvrkwjpaqaxmaubiysrazakoxqegckybcttwpscddovmossbb";//
            String str = in.readLine();
            //String res = "sqbnyzclwqawbpdtzincfawrwlfmefhatqeedicyvrkwjpaqaxmaubiysrazakoxqegckybcttwpscddovmsbbos";
            String res = out.readLine();
            int[] pos;
            if(str.length() != 1){
                pos = lex(str);
                 //System.out.printf("j : %d -- k : %d\n", pos[0], pos[1]);
                if(pos[0] != -1){
                    //System.out.printf("j : %d -- k : %d\n", pos[0], pos[1]);
                result(str, pos[0], pos[1], res);
                }
                else System.out.println("no answer");
            }
            else
                System.out.println("no answer");
        }
    }
}