\ Model demo_20260112_163742
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x4261b57aaaccbcb7
Minimize
  9 assign[I,A] + 4 assign[I,B] + 3 assign[I,C] + 7 assign[I,D]
   + 4 assign[II,A] + 6 assign[II,B] + 5 assign[II,C] + 6 assign[II,D]
   + 5 assign[III,A] + 4 assign[III,B] + 7 assign[III,C] + 5 assign[III,D]
   + 7 assign[IV,A] + 5 assign[IV,B] + 2 assign[IV,C] + 3 assign[IV,D]
   + 10 assign[V,A] + 6 assign[V,B] + 7 assign[V,C] + 4 assign[V,D]
Subject To
 task_assignment__t_A: assign[I,A] + assign[II,A] + assign[III,A]
   + assign[IV,A] + assign[V,A] = 1
 task_assignment__t_B: assign[I,B] + assign[II,B] + assign[III,B]
   + assign[IV,B] + assign[V,B] = 1
 task_assignment__t_C: assign[I,C] + assign[II,C] + assign[III,C]
   + assign[IV,C] + assign[V,C] = 1
 task_assignment__t_D: assign[I,D] + assign[II,D] + assign[III,D]
   + assign[IV,D] + assign[V,D] = 1
 worker_limit__w_I: assign[I,A] + assign[I,B] + assign[I,C] + assign[I,D]
   <= 1
 worker_limit__w_II: assign[II,A] + assign[II,B] + assign[II,C]
   + assign[II,D] <= 1
 worker_limit__w_III: assign[III,A] + assign[III,B] + assign[III,C]
   + assign[III,D] <= 1
 worker_limit__w_IV: assign[IV,A] + assign[IV,B] + assign[IV,C]
   + assign[IV,D] <= 1
 worker_limit__w_V: assign[V,A] + assign[V,B] + assign[V,C] + assign[V,D]
   <= 1
 select_4_workers: assign[I,A] + assign[I,B] + assign[I,C] + assign[I,D]
   + assign[II,A] + assign[II,B] + assign[II,C] + assign[II,D]
   + assign[III,A] + assign[III,B] + assign[III,C] + assign[III,D]
   + assign[IV,A] + assign[IV,B] + assign[IV,C] + assign[IV,D]
   + assign[V,A] + assign[V,B] + assign[V,C] + assign[V,D] = 4
Bounds
Binaries
 assign[I,A] assign[I,B] assign[I,C] assign[I,D] assign[II,A] assign[II,B]
 assign[II,C] assign[II,D] assign[III,A] assign[III,B] assign[III,C]
 assign[III,D] assign[IV,A] assign[IV,B] assign[IV,C] assign[IV,D]
 assign[V,A] assign[V,B] assign[V,C] assign[V,D]
End
