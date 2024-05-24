# VTL
Visual transductive learning via iterative label correction
This code implements Visual transductive learning via iterative label correction (VTL), which is published at Multimedia Systems (2024). You can download the paper from : "https://doi.org/10.1007/s00530-024-01339-3". 
# Abstract
Unsupervised domain adaptation (UDA) aims to transfer knowledge across domains when there is no labeled data available inthe target domain. In this way, UDA methods attempt to utilize pseudo-labeled target samples to align distribution across thesource and target domains. To mitigate the negative impact of inaccurate pseudo-labeling on knowledge transfer, we proposea novel UDA method known as visual transductive learning via iterative label correction (VTL) that benefits from a novellabel correction paradigm. Specifically, we learn a domain invariant and class discriminative latent feature representationfor samples of both source and target domains. Simultaneously, VTL attempts to locally align the information of samples bylearning the geometric structure of samples. Therefore, the novel label correction paradigm is utilized on embedded sharedfeature space to effectively maximize the reliability of pseudo labels by correcting inaccurate pseudo-labeled target samples,which significantly improves VTL classification performance. The experimental results on several object recognition tasksverified our proposed VTL superiority in comparison with other state-of-the-arts.

# Run
The original code is implemented using Matlab R2016a. For running the code, create a folder "data/surf" wich includes surf data .mat files and then run the "main.m" file.

# References
Rezaei, S., Ahmadvand, M. and Tahmoresnezhad, J., 2024. Visual transductive learning via iterative label correction. Multimedia Systems, 30(3), p.145.
