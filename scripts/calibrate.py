from grounded_retrieval import ChunkStore, SentenceTransformerBackend, GroundedRetriever
from grounded_retrieval.leash import LeashPolicy, query_term_coverage, SupportedSpan
b = SentenceTransformerBackend(); r = GroundedRetriever(ChunkStore(), b)

ANSWERABLE = [
 "what happens when an agent violates an authority boundary","exit code 2",
 "how does the BPMN model fail closed","Deployment Envelope Hash",
 "what is autonomy budget preemption","how are receipts verified offline",
 "what is the toxic topology detector","how does the policy engine refuse",
 "what tests cover the governance sandbox","what is a completion receipt",
 "how does spine lite integrate with claude code","what is proposal versus execution",
]
UNANSWERABLE = [
 "what is the best pizza topping in Naples","how do I configure Kubernetes autoscaling for this",
 "what is the SOC 2 certification status","who won the 2024 world series",
 "what is the refund policy","what is the OAuth device flow implementation",
 "how much does the enterprise license cost","what is the GDPR data retention period",
 "which CVE numbers were assigned","what is the on-call rotation schedule",
]
def probe(qs):
    out=[]
    for q in qs:
        res = r.retrieve(q)
        sims=[x.dense_similarity for x in res if x.dense_similarity is not None]
        top=max(sims) if sims else 0.0
        spans=tuple(SupportedSpan(x.chunk.chunk_id,x.chunk.locator(),x.dense_similarity or 0,x.chunk.text) for x in res[:5])
        out.append((q, top, query_term_coverage(q, spans)))
    return out
A=probe(ANSWERABLE); U=probe(UNANSWERABLE)
print("ANSWERABLE  sim range: %.3f - %.3f | cov range %.2f - %.2f" % (min(x[1] for x in A),max(x[1] for x in A),min(x[2] for x in A),max(x[2] for x in A)))
print("UNANSWERABLE sim range: %.3f - %.3f | cov range %.2f - %.2f" % (min(x[1] for x in U),max(x[1] for x in U),min(x[2] for x in U),max(x[2] for x in U)))
print()
best=None
for sim_t in [round(0.40+0.01*i,2) for i in range(45)]:
    for cov_t in [0.0,0.2,0.3,0.4,0.5,0.6,0.7]:
        ans_ok = sum(1 for q,s,c in A if s>=sim_t and c>=cov_t)
        ref_ok = sum(1 for q,s,c in U if not(s>=sim_t and c>=cov_t))
        # balanced: answer the answerable, refuse the unanswerable
        score = ans_ok/len(A) + ref_ok/len(U)
        if best is None or score>best[0]: best=(score,sim_t,cov_t,ans_ok,ref_ok)
print("BEST: sim>=%.2f cov>=%.2f -> answered %d/%d answerable, refused %d/%d unanswerable (score %.3f)"%(best[1],best[2],best[3],len(A),best[4],len(U),best[0]))
print()
print("--- per-query at best thresholds ---")
for q,s,c in A:
    print("  A %s sim=%.3f cov=%.2f  %s"%("OK " if (s>=best[1] and c>=best[2]) else "MISS",s,c,q[:55]))
for q,s,c in U:
    print("  U %s sim=%.3f cov=%.2f  %s"%("OK " if not(s>=best[1] and c>=best[2]) else "LEAK",s,c,q[:55]))
