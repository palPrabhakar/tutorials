import java.util.Iterator;
import edu.princeton.cs.algs4.*;

public class Parantheses {
	
	public static void main(String[] args) {
		
		Stack<String> stack = new Stack<String>();
		
		while(!StdIn.isEmpty()){
			String s = StdIn.readString();
			if(s.equals("{")) stack.push(s);
			else if(s.equals("[")) stack.push(s);
			else if(s.equals("(")) stack.push(s);
			else if(s.equals("}")) {
				String temp = stack.pop();
				StdOut.println(temp); 
				if(!temp.equals("{")) { 
					StdOut.println("Wrong Order"); 
					//break;
					}
			}
			else if(s.equals("]")){
				String temp = stack.pop();
				StdOut.println(temp); 
				if(!temp.equals("[")) { 
					StdOut.println("Wrong Order"); 
					//break;
					}
			}
			else{
				String temp = stack.pop();
				StdOut.println(temp);
				if(!temp.equals("(")) {  
					StdOut.println("Wrong Order");
					//break;
					}
			}
			
		}
		
		if(!stack.isEmpty()) StdOut.println("No of right Parantheses less than left Paranthese");
		
	}

}
