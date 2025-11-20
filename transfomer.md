###### Transformer Model  

```  
                                                  Infer only, Train & Eval are unneeded.  
 |                                                       → → → → → → → → → → → → → → ↓  
 |                                                       ↑                           ↓  
 |          |                                           x_tgt(porb)   |          |   ↓  
 |          |                                            ↑            |          |   ↓  
-------------------------------------------------------- ↑ ------------------------  ↓  
 |          |                                            ↑            |          |   ↓  
 |  output  |                                           softmax       |  output  |   ↓  
 |   layer  |                                            ↑            |  layer   |   ↓  
 |          |                                            linear       |          |   ↓  
 |          |                                            ↑            |          |   ↓  
-------------------------------------------------------- ↑ ------------------------  ↓  
 |          |                                            ↑            |          |   ↓  
 |          |                                layer normalization ← ←  |          |   ↓  
 |          |                                            ↑         ↑  |          |   ↓  
 |          |                               feed forward network   ↑  |          |   ↓  
 |          |                                            ↑         ↑  |          |   ↓  
 |          |            → → → → → → → → ↓               ↑ → → → → ↑  |          |   ↓  
 |          |            ↑               ↓               ↑            |          |   ↓  
 |  encode  |  → → layer normalization   ↓   layer normalization ← ←  |  encode  |   ↓  
 |   layer  |  ↑         ↑               ↓               ↑         ↑  |  layer   |   ↓  
 |          |  ↑   feed forward network  ↓  multi-head attention   ↑  |          |   ↓  
 |    x     |  ↑         ↑               ↓       ↑   ↑   ↑         ↑  |     x    |   ↓  
 |          |  ↑ ← ← ← ← ↑               → → → → + → ↑   ↑ → → → → ↑  |          |   ↓  
 |    n     |            ↑                               ↑            |     n    |   ↓  
 |          |  → → layer normalization       layer normalization ← ←  |          |   ↓  
 |          |  ↑         ↑                               ↑         ↑  |          |   ↓  
 |          |  ↑   multi-head attention     multi-head attention   ↑  |          |   ↓  
 |          |  ↑     ↑   ↑   ↑                       ↑   ↑   ↑     ↑  |          |   ↓  
 |          |  ↑ ← ← ↑ ← + → ↑                       ↑ ← + → → → → ↑  |          |   ↓  
 |          |            ↑                               ↑            |          |   ↓  
------------------------ ↑ ----------------------------- ↑ ------------------------  ↓  
 |          |            ↑                               ↑            |          |   ↓  
 |   input  |      positional encoding       positional encoding      |  input   |   ↓  
 |   layer  |            ↑                               ↑            |  layer   |   ↓  
 |          |      token embedding               token embedding      |          |   ↓  
 |          |            ↑                               ↑            |          |   ↓  
------------------------ ↑ ----------------------------- ↑ ------------------------  ↓  
 |          |            ↑                               ↑            |          |   ↓  
 |          |           x_src                           x_tgt(shft)   |          |   ↓  
 |                                                       ↑                           ↓  
 |                                                       ↑ ← ← ← ← ← ← ← ← ← ← ← ← ← ←  
                                                  Infer only, Train & Eval are unneeded.  
```  


