\ Model rebuilt_from_json
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xe3bad98c830ea3ba
Minimize
  0 assign[I,A] + 0 assign[I,B] + 0 assign[I,C] + 0 assign[I,D]
   + 0 assign[II,A] + 0 assign[II,B] + 0 assign[II,C] + 0 assign[II,D]
   + 0 assign[III,A] + 0 assign[III,B] + 0 assign[III,C] + 0 assign[III,D]
   + 0 assign[IV,A] + 0 assign[IV,B] + 0 assign[IV,C] + 0 assign[IV,D]
   + 0 assign[V,A] + 0 assign[V,B] + 0 assign[V,C] + 0 assign[V,D]
Subject To
 task_A_assigned: = 1
 task_B_assigned: = 1
 task_C_assigned: = 1
 task_D_assigned: = 1
 worker_I_assigned: <= 1
 worker_II_assigned: <= 1
 worker_III_assigned: <= 1
 worker_IV_assigned: <= 1
 worker_V_assigned: <= 1
 four_workers_assigned: = 4
Bounds
Binaries
 assign[I,A] assign[I,B] assign[I,C] assign[I,D] assign[II,A] assign[II,B]
 assign[II,C] assign[II,D] assign[III,A] assign[III,B] assign[III,C]
 assign[III,D] assign[IV,A] assign[IV,B] assign[IV,C] assign[IV,D]
 assign[V,A] assign[V,B] assign[V,C] assign[V,D]
End
