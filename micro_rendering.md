

<!-- Start of picture text -->
_— ee<br>Ni<br>eee en |,<br>eera<br>Aa Saytg i be —<br>(4|Tae1re MaleSeOat32223eelgAgsaad : NnnityBaBN . ; oe =<br>iti ti aea RS.Ceeee Wu / asri ; Ldy 4aGia | me = =:~~>——. . .<br>.<br>il 1 32522 5,” Diffuse<br><!-- End of picture text -->

132:2       •       T. Ritschel et al. 

## **1.1 Related Work** 

We briefly review related work, with a focus on final gathering and global illumination (GI). 

**Final Gathering** Final gathering refers to the process of calculating the amount of indirect illumination at a surface point. For instance, it is an integral part of photon mapping [Jensen 1996]. It is usually the most time consuming part in GI and, not surprisingly, many algorithms aim at reducing the final gathering cost. 

Irradiance caching [Ward et al. 1988] performs final gathering only at a sparse set of locations in the image, and interpolates irradiance values at other locations. Irradiance gradients [Ward and Heckbert 1992] additionally compute the gradient of the irradiance to enable better interpolation. Kˇrivánek et al. [2005] extended irradiance caching to glossy surfaces by storing incident radiance instead of irradiance. Our micro-rendering technique speeds up the final gathering process itself and is independent of the cache placement strategy. 

The lightcuts method [Walter et al. 2005] pursues a similar goal, but assumes that all (indirect) lighting is represented as a hierarchical collection of point lights. For every visible surface point, a cut through the hierarchy is computed, and the contributions of all clustered point lights are summed up; visibility is resolved by raycasting toward the center of each cluster. In contrast, our method uses a hierarchy of point samples, which we rasterize in parallel into micro-buffers at less cost while enabling more accurate visibility. Recently, lightcuts were also extended to combine several cuts through a hierarchy of lights, visibility, and BRDFs [CheslackPostava et al. 2008]. While this extension allows for interactive performance, it does assume static geometry. Matrix row-column sampling [Hašan et al. 2007] computes a single set of point lights instead of varying it per image location. However, importance sampling is not possible and rendering quality is limited due to the low number of gathering samples used. Recently, Christensen [2008] proposed a CPU-based method to speed up final gathering for diffuse and moderately glossy scenes using a point-based representation of direct illumination stored in an octree. At each irradiance cache location, distant points are rasterized into a cube map and nearby points are raycast, but since no importance warping is used, glossiness is directly limited by the buffer’s resolution. Our method follows a similar goal: speeding up global illumination through a hierarchical representation of illumination. In contrast to Christensen [2008], our method exploits GPU compute power and employs importance sampling for arbitrary BRDFs – both for rasterization and on-demand raycasting. In concurrent work, Wang et al. [2009] demonstrate interactive global illumination using GPUbased final gathering with ray-tracing. It enables complex lighting effects but relies on sparse gathering locations for efficiency. 

**Real-time GI** Real-time global illumination for static scenes is possible with a number of different techniques. One of the early methods is precomputed radiance transfer (PRT) [Sloan et al. 2002]. Most of the PRT variants require static geometry, although some recent extensions also allow the movement of rigid objects [Iwasaki et al. 2007]. Low-resolution dynamic scenes are possible, assuming low-frequency illumination [Ren et al. 2006; Sloan et al. 2007]. Recently, Lehtinen et al. [2008] presented an interactive PRT-based illumination method for static scenes using a hierarchical, pointbased representation. Light transport was simulated in a similar manner to radiosity [Cohen and Wallace 1993]. We also employ a hierarchical point-based representation but only for emitters; receiver points are selected in image-space and gathering is only performed for those. 

**Interactive GI in Dynamic Scenes** Interactive frame rates for GI in moderately complex scenes can be achieved using antiradiance [Dachsbacher et al. 2007], but computing a link hierarchy for fully dynamic scenes requires additional work [Meyer et al. 

2009]. Bunnel [2005] approximated ambient occlusion and indirect illumination in small scenes using a hierarchy of linked disc elements. Instant radiosity [Keller 1997] is the basis for many interactive GI approaches but is too slow for complex scenes in its basic form. However, it can be efficiently implemented on GPUs using reflective shadow maps [Dachsbacher and Stamminger 2005; Dachsbacher and Stamminger 2006] when indirect visibility is ignored. Imperfect shadow maps [Ritschel et al. 2008] achieve interactive frame rates for moderately complex and fully dynamic scenes using approximate visibility, but ultimately fail to handle large scenes due to a non-hierarchical point representation. Further, indirect shadows are generally smoothed out considerably. In contrast, our approach enables high-quality, indirect illumination for both diffuse and glossy scenes of high geometric complexity. 

**Discussion** Most of the above techniques share the goal of evaluating the rendering equation more densely, when one of the factors inside the integral is high, and less densely everywhere else. This goal is also the inspiration of our technique but in contrast to many of the above techniques we evaluate everything on the fly. Traditional importance sampling in ray tracing [Dutré et al. 2006] is probably the best-known method that tries to focus computation on where it is most needed. It has usually been coupled with ray tracing, since visibility needs to be checked in arbitrary directions. We make this idea amenable to GPU-based rendering by rasterizing our hierarchy of points into warped micro-buffers, effectively performing importance sampling. 

## **1.2 Overview** 

Our method allows the rendering of global illumination in fully dynamic scenes. It is scalable such that the user can trade rendering quality for speed, with a smooth transition ranging from fast previews to solutions that are close to ground truth. At the heart of our method is an efficient micro-rendering technique which performs a BRDF importance sampled final gathering of the incident radiance. The term “micro” refers to the low overhead of launching the rendering, as well as to the low resolution of the frame buffer. We demonstrate that our method runs an order of magnitude faster than previous approaches and reaches preview quality at interactive speeds of up to 10 frames per second. The key points of our algorithm are: 

- We generate a hierarchical point-based representation of the scene’s surfaces for adaptive level-of-detail rendering. 

- Our novel micro-rendering technique facilitates a highly parallel rendering of arbitrary (in our case hemispherical) mappings on the GPU, in order to gather the incident radiance at many surface points at the same time. 

- We integrate importance sampling into the micro-rendering allowing us to efficiently compute final gathering for diffuse and glossy surfaces with arbitrary BRDFs. 

- For preview quality we compute final gathering at a subset of the image pixels and use bilateral upsampling [Sloan et al. 2007]. High-quality renderings at approximately 0.5 to 1 frames per second perform final gathering at every pixel (512 _×_ 512 res.). 

Micro-rendering is beneficial for many different global illumination methods. It directly renders one-bounce indirect illumination when point samples are directly lit, and multiple bounces when used with instant radiosity techniques. It can also be used to compute and display radiosity solutions, and for interactive walkthroughs of photon mapping results with final gathering. 

# **2 Scalable, Parallel Final Gathering** 

In this section we describe all steps of the micro-rendering method in detail, starting with the basic technique, which is then extended with importance sampling and bilateral upsampling. 

ACM Transactions on Graphics, Vol. 28, No. 5, Article 132, Publication date: December 2009. 

Micro-Rendering for Scalable, Parallel Final Gathering       •       132:3 



<!-- Start of picture text -->
G<br><!-- End of picture text -->







<!-- Start of picture text -->
E<br><!-- End of picture text -->



<!-- Start of picture text -->
F<br><!-- End of picture text -->



<!-- Start of picture text -->
G<br><!-- End of picture text -->



<!-- Start of picture text -->
A<br><!-- End of picture text -->



<!-- Start of picture text -->
B<br><!-- End of picture text -->



<!-- Start of picture text -->
C<br><!-- End of picture text -->



<!-- Start of picture text -->
D<br><!-- End of picture text -->

**_Figure 2:_** _We represent the scene’s surfaces using a point hierarchy, similar to QSplat, which is stored as a complete binary tree. This allows for easy traversal and fast updates at run-time._ 

## **2.1 Hierarchical Point-Based Representation** 

Our micro-rendering method is based on a hierarchical point-based representation of the scene, since point-based representations allow for efficient level-of-detail rendering on GPUs [Dachsbacher et al. 2003; Ritschel et al. 2008], often provide a simple point selection criterion [Rusinkiewicz and Levoy 2000], and have shown to be well-suited for approximative global illumination [Christensen 2008]. In comparison to triangle-based rendering, point-based rendering has a lower setup and rasterization cost for low image resolutions. Similar to QSplat, we use a hierarchy of bounding spheres, where leaf nodes represent a single surface element (an oriented disc with radius _r_ ) and interior nodes represent a collection of surface elements. For rendering, this hierarchy is traversed starting from the root node: each node’s bounding sphere is projected into micro screen space to compare its size to a given threshold. This determines if the respective disc is rendered and the traversal is terminated, or if its child-nodes are to be tested recursively. We replace the ‘size-in-screen-space’-test of the original QSplat method with a test based on the solid-angle subtended by a node. This criterion will allow us to define _warped projective mappings_ , and thus to integrate importance sampling easily. Note that warped mappings would be difficult to combine with triangle rasterization. 

**Point Hierarchy Generation** The point-based representation of the scene is generated in an offline preprocessing step. First, we create random points on the triangles of the scene, proportional to the triangle areas, using a best candidate sampling. For every point sample, we store the triangle index and the barycentric coordinates of the point relative to its triangle. The barycentric coordinates allow us to recompute positions and normals for deforming geometry [Ritschel et al. 2008]. The point density, which is initially constant, determines the radius of the point samples. Under deformations we scale the points’ radii to compensate for the varying point densities. 

These point samples form the leaf nodes of our hierarchical point representation (Fig. 2). We build the hierarchy by computing a binary-space partitioning of the point samples, which we store as a complete binary tree (hence the number of points _n_ is a power of two). This enables us to compute skip-pointers on-the-fly during traversal, instead of storing additional offsets. The construction first sorts the leaf nodes and works as follows: we take the list of leaf nodes (the initially created point samples) as input and determine along which coordinate axis the point list has the largest extent. We then sort all points along this axis, split the list into two parts with an equal number of points, and recursively process both sublists in the same manner. In total, the cost for sorting all points is _O_ ( _n_ log<sup>2</sup> _n_ ) for _n_ points. As we use a complete tree, the order of the points in the list implicitly defines the hierarchy. Consider the example shown in Fig. 2: nodes A to D are the leaf nodes after sorting. As the tree is complete, nodes A and B are children of the node E, and so on. For all interior nodes, we compute the minimum bounding sphere enclosing all child nodes, as well as the cone of normals (stored as direction plus cone angle). 



<!-- Start of picture text -->
F<br>B<br>E A G<br><!-- End of picture text -->



<!-- Start of picture text -->
Ω (x3,y3) B<br>E A<br>Ф(x2,y2)<br>Ф(x1,y1)<br><!-- End of picture text -->

**_Figure 3:_** _Every pixel_ ( _xi, yi_ ) _of a micro-buffer corresponds to a direction_ Φ( _xi, yi_ ) _and subtends a solid angle_ Ω( _xi, yi_ ) _. The point hierarchy is traversed and rasterized such that nodes project to no more than one pixel in the micro-buffer. In this example, the nodes A, B, and F, of the point hierarchy in Fig. 2, are selected._ 

**Deforming and Moving Geometry** For deforming geometry, we leave the hierarchy itself unchanged and only update the per-node data. At run-time at the beginning of every frame, we recompute the leaf nodes’ positions and normals, and update the interior nodes, i.e., we recompute the minimum bounding sphere and cone of normals, containing the two child-nodes’ bounding spheres and normals, respectively. This process works bottom-up by successively merging two nodes at a time, yielding a total of _O_ ( _n_ ) operations for _n_ leaves. This keeps the run-time cost for maintaining the point hierarchy low, and we can reasonably handle deforming geometry. For moving objects we create separate point hierarchies. The normals and normal cones are used for lighting computation and back-face culling during the point hierarchy traversal. 

## **2.2 Final Gathering Using Micro-Rendering** 

Final gathering is used for high-quality renderings to compute the indirect illumination at every visible surface point **p** . It involves gathering incident radiance _L_ in( _ωi_ ) from direction _ωi_ of the upper hemisphere at **p** . Usually BRDF importance sampling is used to gather more from directions that contribute more to the reflected radiance towards the observer. Due to the typically large number of gather directions involved, this is an expensive operation and commonly used in the context of offline rendering only. Our method enables parallel final gathering and achieves interactive frame rates through micro-rendering, which has been developed with the high parallelism of contemporary and future GPUs in mind. In the following we first detail the basic micro-rendering procedure for a single gather point. In Section 3 we then describe the implementation details, and how we ensure that the computational power of such hardware is utilised to a very high degree. 

Micro-rendering generates images using the mapping Φ( _x, y_ ) = _ω_ relating a pixel ( _x, y_ ) of the micro-buffer to a gather direction _ω_ . We denote the solid angle subtended by the pixel under this mapping as Ω( _x, y_ ) (see Fig. 3). Such a mapping can be any standard hemispherical projection; however, we use our own mapping as described in the next subsection. The micro-buffers store an index of the nearest visible node as well as its distance at every pixel (i.e., we maintain an index and depth buffer); A micro-buffer is typically small, ranging from 8 _×_ 8 to 24 _×_ 24 pixels in our examples. 

The basic image formation process starts with computing the cut in the point hierarchy, which also determines which point sample is visible for every micro-pixel. We then gather the incident radiance for every micro-pixel, and convolve it with the BRDF, yielding the radiance reflected towards the observer. Importance sampling can be integrated easily at little additional cost by changing the mapping function Φ( _x, y_ ) appropriately (Section 2.3). 

**Point Hierarchy Cut** We compute the cut using a depth-first search in the point hierarchy starting from the root node. For each node, we evaluate the selection criterion: we first compute 

ACM Transactions on Graphics, Vol. 28, No. 5, Article 132, Publication date: December 2009. 





<!-- Start of picture text -->
.<br><!-- End of picture text -->



<!-- Start of picture text -->
=<br><!-- End of picture text -->



<!-- Start of picture text -->
|<br>;<br><!-- End of picture text -->

( ~~v”~*™~~ 

~~—~~ 

J 







<!-- Start of picture text -->
2 . as ath > ar meee i<br>ai<br><!-- End of picture text -->



<!-- Start of picture text -->
| ISM it Micros Refihetne<br>- 3) OMe oF<br>* ig<br>Pe a ae<br><!-- End of picture text -->



<!-- Start of picture text -->
Ss |<br>iff<br>bh : ao J 7 om.<br>k=0,N = 20 kj=0'5,/N =5 =!<br><!-- End of picture text -->



<!-- Start of picture text -->
Without r F With F.<br>ray casting - ray casting<br>“abe<br>ay. y<br><!-- End of picture text -->

Micro-Rendering for Scalable, Parallel Final Gathering       •       132:7 



**_Figure 12:_** _Influence of micro-buffer size on rendering quality (_ 256 _×_ 256 _). We use_ 8 _×_ 8 _(3.2 Hz),_ 16 _×_ 16 _(1.5 Hz), and_ 24 _×_ 24 _(0.7 Hz). Smaller sizes are faster but quality decreases._ 

more gather samples per second when the total number of gather samples increases. This is to be expected due to the increased coherency between gather samples. 

The computation time of our GPU-based final gathering technique is (roughly) split as follows for a typical scene, such as the teaser. 1.5% is spent on updating the hierarchy, 2% on building the view-dependent per-pixel mappings Φ( _x, y_ ), 18% on evaluating them, 60% on rasterizing the point hierarchy, 8% on ray casting, and 11% on bilateral upsampling, tonemapping and direct lighting. 

## **5.1 Discussion and Limitations** 

In typical scenes, our method is able to compute about 150M final gathering samples with importance sampling (a 512 _×_ 512 image with 24 _×_ 24 micro-buffers at every pixel renders at about 1Hz, including tree update, shading, etc.). CPU-based ray tracing can send out about 10M rays per second per core, if the rays are coherent [Shevtsov et al. 2007], i.e., about an order of magnitude less than our method. Currently reported numbers on GPU-based ray tracing indicate that about 20M rays can be traced in dynamic scenes (including a complete rebuild of the acceleration structure) [Zhou et al. 2008]. 

While our method deals well with complex scene and arbitrary BRDFs, it has certain limitations. When glossy surfaces are present in a scene, more gather samples are required. In this case our simple regular gather sample distribution is not ideal and more elaborate distributions should be used [Kˇrivánek et al. 2005]. As mentioned earlier, we have opted to prevent banding artifacts by jittering the local coordinate system at each pixel. As a result some noise is visible in our images. While our method renders one-bounce indirect caustics – simply by gathering from highly glossy surfaces – more than two specular bounces are not supported. Other effects, such as transparent objects and refractions are currently not simulated. 

We implemented our method in CUDA to explore the potential of parallelizing the various micro-rendering tasks (see Section 3). In the end, performing all tasks in a single kernel was fastest. This suggests that a pure OpenGL implementation might be just as fast. 

# **6 Conclusions and Future Work** 

We have presented a novel technique for scalable and parallel final gathering that enables the efficient computation of indirect illumination. It has been designed to harness the power of modern GPUs, and can trade rendering quality for computation time in a simple and intuitive manner. It handles large, fully dynamic scenes with diffuse and glossy surfaces. We demonstrate various applications of our method including single and multiple bounce indirect illumination, and the computation and display of radiosity solutions. Further, it can be used to compute final gathering for interactively rendering photon mapping results, yielding a speedup of an order of magnitude over a standard CPU implementation. 

There are several possible avenues for future research. Our method is geared towards diffuse and glossy surfaces; highly specular surfaces require a sufficient number of gather locations as well as point samples to capture the illumination. We would like to investigate alternatives such as radiance caching on the GPU to handle 



<!-- Start of picture text -->
tree update rendering 10³ samples/sec with space-flling curve<br>35 4 10³ samples/sec without space flling curve<br>30 3,5 400<br>25 3 300<br>20 2,5<br>2<br>15 200<br>1,5<br>10 1 100<br>5 0,5<br>0 0 0<br>10 15 20 0 250 500 750 1000<br>tree depth final gather samples (10³ samples)<br>me  (ms) t<br>t me  (sec)<br>10³ samples / sec<br><!-- End of picture text -->

**_Figure 13:_** _Left: the red curve indicates computation time vs. scene complexity (measured as tree depth, equals_ 2<sup>_N_</sup> _point samples). It indicates sub-linear complexity. The blue curve outlines treeupdating time. Right: the number of processed gather samples per second vs. the total number of gather samples in the image (green with, blue without space-filling curve to support spatial coherence)._ 

specular surfaces. Our hierarchical point representation allows for large and complex geometry; however, scenes with a high depth complexity, such as buildings with many rooms, would benefit from portal decompositions and efficient culling techniques. 

# **References** 

- BUNNELL, M. 2005. Dynamic ambient occlusion and indirect lighting. In _GPU Gems 2_ , M. Pharr, Ed. Add. Wesley, 223–233. 

- CHESLACK-POSTAVA, E., WANG, R., AKERLUND, O., AND PELLACINI, F. 2008. Fast, realistic lighting and material design using nonlinear cut approximation. _ACM Trans. Graph. (Proc. SIGGRAPH Asia) 27_ , 5, 128:1–128:10. 

- CHRISTENSEN, P. 2008. Point-based approximate color bleeding. Tech. Rep. 08-01, Pixar Animation Studios. 

- COHEN, M., AND WALLACE, J. 1993. _Radiosity and Realistic Image Synthesis_ . Academic Press Professional. 

- DACHSBACHER, C., AND STAMMINGER, M. 2005. Reflective shadow maps. In _Proc. I3D_ , 203–213. 

- DACHSBACHER, C., AND STAMMINGER, M. 2006. Splatting indirect illumination. In _Proc. I3D_ , 93–100. 

- DACHSBACHER, C., VOGELGSANG, C., AND STAMMINGER, M. 2003. Sequential point trees. _ACM Trans. Graph. (Proc. SIGGRAPH) 22_ , 3, 657–662. 

- DACHSBACHER, C., STAMMINGER, M., DRETTAKIS, G., AND DURAND, F. 2007. Implicit visibility and antiradiance for interactive global illumination. _ACM Trans. Graph. (Proc. SIGGRAPH) 26_ , 3. 

- DUTRÉ, P., BALA, K., AND BEKAERT, P. 2006. _Advanced Global Illumination_ . AK Peters. 

- HAŠAN, M., PELLACINI, F., AND BALA, K. 2007. Matrix row-column sampling for the many-light problem. _ACM Trans. Graph. (Proc. SIGGRAPH) 26_ , 3, 26. 

- IWASAKI, K., DOBASHI, Y., YOSHIMOTO, F., AND NISHITA, T. 2007. Precomputed radiance transfer for dynamic scenes taking into account light interreflection. In _Proc. EGSR_ , 35–44. 

- JENSEN, H. W. 1995. Importance driven path tracing using the photon map. In _Proc. ESGR_ , 326–335. 

- JENSEN, H. W. 1996. Global illumination using photon maps. In _Proc. EGSR_ , 21–30. 

- KELLER, A. 1997. Instant radiosity. In _SIGGRAPH ’97_ , 49–56. 

- KRIVÁNEK<sup>ˇ</sup> , J., GAUTRON, P., PATTANAIK, S., AND BOUATOUCH, K. 2005. Radiance caching for efficient global illumination computation. _IEEE TVCG 11_ , 5, 550–561. 

- LEHTINEN, J., ZWICKER, M., TURQUIN, E., KONTKANEN, J., DURAND, F., SILLION, F., AND AILA, T. 2008. A meshless hierarchical representation for light transport. _ACM Trans. Graph. (Proc. SIGGRAPH) 27_ , 3, 37:1–37:9. 

ACM Transactions on Graphics, Vol. 28, No. 5, Article 132, Publication date: December 2009. 



**_Figure 14:_** _Comparison between 1. Fast preview images (_<sup>_1_</sup> _/16 resolution and upsampling), 2. Non-filtered micro-rendering for all_ 512 _×_ 512 _pixels and 3. PBRT path tracing. The simple Cornell box achieves high-quality global illumination results even with bilateral upsampling. The geometrically complex plant scene shows some slight differences (see insets), which we attribute to the discrete micro-buffers. For the Sponza scene, our method produces results that are indistinguishable from the reference rendering. In fact, bilateral upsampling removes noise and produces the visually most pleasing result. We also achieve very similar results for the glossy scene. However, as expected, bilateral upsampling changes the glossy reflection on the sphere slightly. The error images are scaled by a factor of three._ 

- MEYER, Q., EISENACHER, C., STAMMINGER, M., AND DACHSBACHER, C. 2009. Data-parallel hierarchical link creation for radiosity. In _Proc. EGPGV_ , 65–70. 

- PHARR, M., AND HUMPHREYS, G. 2004. _Physically Based Rendering: From Theory to Implementation_ . Morgan Kaufmann. 

- REN, Z., WANG, R., SNYDER, J., ZHOU, K., LIU, X., SUN, B., SLOAN, P.-P., BAO, H., PENG, Q., AND GUO, B. 2006. Realtime soft shadows in dynamic scenes using spherical harmonic exponentiation. _ACM Trans. Graph. (Proc. SIGGRAPH) 25_ , 3, 977–986. 

- RITSCHEL, T., GROSCH, T., KIM, M. H., SEIDEL, H.-P., DACHSBACHER, C., AND KAUTZ, J. 2008. Imperfect shadow maps for efficient computation of indirect illumination. _ACM Trans. Graph. (Proc. SIGGRAPH Asia) 27_ , 5, 129:1–129:8. 

- RUSINKIEWICZ, S., AND LEVOY, M. 2000. QSplat: A multiresolution point rendering system for large meshes. In _Proc. SIGGRAPH_ , 343–352. 

- SHEVTSOV, M., SOUPIKOV, A., AND KAPUSTIN, A. 2007. Highly parallel fast kd-tree construction for interactive ray tracing of dynamic scenes. _Computer Graphics Forum (Proc. Eurographics) 26_ , 3, 395–404. 

- SLOAN, P.-P., KAUTZ, J., AND SNYDER, J. 2002. Precomputed radiance transfer for real-time rendering in dynamic, lowfrequency lighting environments. _ACM Trans. Graph. (Proc. SIGGRAPH) 21_ , 3, 527–536. 

- SLOAN, P.-P., GOVINDARAJU, N., NOWROUZEZAHRAI, D., AND SNYDER, J. 2007. Image-based proxy accumulation for realtime soft global illumination. In _Proc. Pacific Graphics_ , 97–105. 

- WALTER, B., FERNANDEZ, S., ARBREE, A., BALA, K., DONIKIAN, M., AND GREENBERG, D. P. 2005. Lightcuts: A scalable approach to illumination. _ACM Trans. Graph. (Proc. SIGGRAPH) 24_ , 3, 1098–1107. 

- WANG, R., WANG, R., ZHOUN, K., PAN, M., AND BAO, H. 2009. An efficient GPU-based approach for interactive global illumination. _ACM Trans. Graph. (SIGGRAPH) 28_ , 3, 91:1–91:8. 

- WARD, G., AND HECKBERT, P. 1992. Irradiance gradients. In _Proc. EGSR_ , 85–98. 

- WARD, G., RUBINSTEIN, F., AND CLEAR, R. 1988. A ray tracing solution for diffuse interreflection. In _Computer Graphics (Proc. SIGGRAPH)_ , vol. 22, 85–92. 

- ZHOU, K., HOU, Q., WANG, R., AND GUO, B. 2008. Real-time kd-tree construction on graphics hardware. _ACM Trans. Graph. (Proc. SIGGRAPH Asia) 27_ , 5, 126:1–126:11. 

ACM Transactions on Graphics, Vol. 28, No. 5, Article 132, Publication date: December 2009. 

