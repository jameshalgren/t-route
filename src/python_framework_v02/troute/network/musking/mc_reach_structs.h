#ifndef MC_REACH_STRUCTS_H
#define MC_REACH_STRUCTS_H
/*
    C Structures
*/
#include "../reach_structs.h"

typedef struct _MC_Segment{
  long id;
  double dt, dx, bw, tw, twcc, n, ncc, cs, s0;
  double qdp, velp, depthp;
} _MC_Segment;

typedef struct _MC_Reach{
  _MC_Segment* _segments;
  int num_segments;
} _MC_Reach;

#endif //MC_REACH_STRUCTS_H
