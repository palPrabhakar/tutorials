import edu.princeton.cs.algs4.IndexMinPQ;
import edu.princeton.cs.algs4.Queue;

public class EagerPrimMST {
	private boolean marked[];
	private IndexMinPQ<Edge> pq;
	private Queue<Edge> mst;
	private double weight;
	
	public EagerPrimMST(EdgeWeighterGraph g) {
		marked = new boolean[g.v()];
		pq = new IndexMinPQ<Edge>(g.v());
		mst = new Queue<Edge>();
		
		visit(g, 0);
		while(!pq.isEmpty()) {
			weight += pq.minKey().weight();
			mst.enqueue(pq.minKey());
			visit(g, pq.delMin());
		}
	}
	
	public void visit(EdgeWeighterGraph g, int v) {
		marked[v] = true;
		for(Edge e : g.adj(v)) {
			int w = e.other(v);
			if(!marked[w]) {
				if(pq.contains(w)){
					if(e.weight() < pq.keyOf(w).weight()) pq.changeKey(w, e);
				}
				else pq.insert(w, e);
			}
		}
	}
	
	
	public double weight() {
		return weight;
	}
	
	public Iterable<Edge> mst() {
		return mst;
	}
	
}
