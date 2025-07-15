Ideas for making a routing-only datastream. Prioritizing feasibility and believeability.
- Define the network representation that is the target for the computing portion. 
- Abstract network representation to another library (one library for nhd-based networks, e.g., RouteLink; another library for the ngen-type networks, e.g., HF-v2.2). The goal here would be to have the network representation expected by the routing algorithm be somewhat standard and then use the libaries to prepare the data. 
- Peel back the diffusive routing and implement only the muskingum cunge.
- Implement direct USGS NWIS-based "Nudging". (Requires adjustments to Input routines.)
- Build a DA method that does upstream propagation of the corrections (Talk to University of Alabama and BYU post-processing groups.) 
- Build direct NWM-S3 input reading. A yaml file can be developed that controls interaction of the wrapper code with any of: 
  - the AWS s3 buckets with forecast forcings
  - the Google s3 buckets with forecast forcings
  - the AWS s3 Retrospective buckets
  - any inflows computed from models developed for the CIROH datastream

