This is a page for evalution with precision, recall.

## SLM

| model | dataset | da_method | macro_precision | macro_recall |
| --- | --- | --- | --- | --- |
| deberta | app | aeda | 0.6333 | 0.6500 |
|  | app | c2l | 0.6437 | 0.6583 |
|  | app | none | 0.6333 | 0.6500 |
|  | app | ssmba | 0.6322 | 0.6460 |
|  | app | ts | 0.6237 | 0.6500 |
|  | gerrit | aeda | 0.8414 | 0.8785 |
|  | gerrit | c2l | 0.8404 | 0.8583 |
|  | gerrit | none | 0.8619 | 0.8619 |
|  | gerrit | ssmba | 0.7862 | 0.8432 |
|  | gerrit | ts | 0.8721 | 0.8438 |
|  | github | aeda | 0.2344 | 0.3333 |
|  | github | c2l | 0.6058 | 0.5798 |
|  | github | none | 0.7122 | 0.6869 |
|  | github | ssmba | 0.7072 | 0.7444 |
|  | github | ts | 0.7293 | 0.7468 |
|  | jira | aeda | 0.8204 | 0.8161 |
|  | jira | c2l | 0.8288 | 0.8291 |
|  | jira | none | 0.8033 | 0.8463 |
|  | jira | ssmba | 0.7806 | 0.8207 |
|  | jira | ts | 0.8301 | 0.8207 |
|  | so | aeda | 0.8808 | 0.8887 |
|  | so | c2l | 0.8915 | 0.8885 |
|  | so | none | 0.8955 | 0.8890 |
|  | so | ssmba | 0.8760 | 0.8807 |
|  | so | ts | 0.8840 | 0.8883 |
|  | tweets | aeda | 0.8038 | 0.6984 |
|  | tweets | c2l | 0.7193 | 0.6916 |
|  | tweets | none | 0.7296 | 0.6590 |
|  | tweets | ssmba | 0.8245 | 0.6401 |
|  | tweets | ts | 0.7627 | 0.7205 |
|  | tweets_n | aeda | 0.6987 | 0.6828 |
|  | tweets_n | c2l | 0.6780 | 0.6853 |
|  | tweets_n | none | 0.6025 | 0.2908 |
|  | tweets_n | ssmba | 0.5729 | 0.5307 |
|  | tweets_n | ts | 0.6594 | 0.5139 |
|  | tweets_p | aeda | 0.7639 | 0.4081 |
|  | tweets_p | c2l | 0.6257 | 0.5557 |
|  | tweets_p | none | 0.5334 | 0.5756 |
|  | tweets_p | ssmba | 0.5323 | 0.5123 |
|  | tweets_p | ts | 0.5239 | 0.5207 |
|  | AVG | - | 0.7220 | 0.6971 |
| t5 | app | aeda | 0.9767 | 0.8086 |
|  | app | c2l | 0.6262 | 0.6377 |
|  | app | none | 0.3951 | 0.4062 |
|  | app | ssmba | 0.6353 | 0.6583 |
|  | app | ts | 0.6460 | 0.6460 |
|  | gerrit | aeda | 0.8177 | 0.8233 |
|  | gerrit | c2l | 0.8396 | 0.8270 |
|  | gerrit | none | 0.7908 | 0.8156 |
|  | gerrit | ssmba | 0.8158 | 0.8324 |
|  | gerrit | ts | 0.8177 | 0.8662 |
|  | github | aeda | 0.7815 | 0.7121 |
|  | github | c2l | 0.5761 | 0.5799 |
|  | github | none | 0.7911 | 0.7780 |
|  | github | ssmba | 0.7644 | 0.7498 |
|  | github | ts | 0.7943 | 0.7719 |
|  | jira | aeda | 0.8200 | 0.8403 |
|  | jira | c2l | 0.8340 | 0.8229 |
|  | jira | none | 0.8301 | 0.8357 |
|  | jira | ssmba | 0.8331 | 0.8285 |
|  | jira | ts | 0.8332 | 0.8342 |
|  | so | aeda | 0.8791 | 0.8861 |
|  | so | c2l | 0.8880 | 0.8931 |
|  | so | none | 0.8892 | 0.8929 |
|  | so | ssmba | 0.8869 | 0.8910 |
|  | so | ts | 0.8825 | 0.8890 |
|  | tweets | aeda | 0.7796 | 0.7160 |
|  | tweets | c2l | 0.5178 | 0.5612 |
|  | tweets | none | 0.7356 | 0.6869 |
|  | tweets | ssmba | 0.7806 | 0.7578 |
|  | tweets | ts | 0.7482 | 0.7285 |
|  | tweets_n | aeda | 0.6265 | 0.5885 |
|  | tweets_n | c2l | 0.5359 | 0.5339 |
|  | tweets_n | none | 0.5222 | 0.5181 |
|  | tweets_n | ssmba | 0.5204 | 0.4924 |
|  | tweets_n | ts | 0.5660 | 0.5365 |
|  | tweets_p | aeda | 0.5101 | 0.5226 |
|  | tweets_p | c2l | 0.6682 | 0.6266 |
|  | tweets_p | none | 0.3824 | 0.4561 |
|  | tweets_p | ssmba | 0.3666 | 0.4762 |
|  | tweets_p | ts | 0.4804 | 0.5709 |
|  | AVG | - | 0.7096 | 0.7075 |
| xlnet | app | aeda | 0.6097 | 0.6173 |
|  | app | c2l | 0.6460 | 0.6460 |
|  | app | none | 0.5957 | 0.6170 |
|  | app | ssmba | 0.6754 | 0.7262 |
|  | app | ts | 0.6353 | 0.6583 |
|  | gerrit | aeda | 0.7988 | 0.8423 |
|  | gerrit | c2l | 0.8151 | 0.8122 |
|  | gerrit | none | 0.8000 | 0.8106 |
|  | gerrit | ssmba | 0.3812 | 0.5000 |
|  | gerrit | ts | 0.8538 | 0.7533 |
|  | github | aeda | 0.7553 | 0.7792 |
|  | github | c2l | 0.5626 | 0.5520 |
|  | github | none | 0.6970 | 0.7220 |
|  | github | ssmba | 0.6765 | 0.6913 |
|  | github | ts | 0.6938 | 0.7603 |
|  | jira | aeda | 0.8195 | 0.8172 |
|  | jira | c2l | 0.8153 | 0.8332 |
|  | jira | none | 0.8340 | 0.8323 |
|  | jira | ssmba | 0.2247 | 0.3333 |
|  | jira | ts | 0.8290 | 0.8218 |
|  | so | aeda | 0.8841 | 0.8804 |
|  | so | c2l | 0.8573 | 0.8625 |
|  | so | none | 0.8841 | 0.8875 |
|  | so | ssmba | 0.8501 | 0.8480 |
|  | so | ts | 0.8793 | 0.8867 |
|  | tweets | aeda | 0.7525 | 0.6241 |
|  | tweets | c2l | 0.6961 | 0.6711 |
|  | tweets | none | 0.2438 | 0.2724 |
|  | tweets | ssmba | 0.6495 | 0.5980 |
|  | tweets | ts | 0.8022 | 0.7671 |
|  | tweets_n | aeda | 0.6096 | 0.5107 |
|  | tweets_n | c2l | 0.6933 | 0.5565 |
|  | tweets_n | none | 0.6888 | 0.4800 |
|  | tweets_n | ssmba | 0.5823 | 0.2954 |
|  | tweets_n | ts | 0.6102 | 0.6087 |
|  | tweets_p | aeda | 0.3904 | 0.4475 |
|  | tweets_p | c2l | 0.4811 | 0.4598 |
|  | tweets_p | none | 0.5090 | 0.5040 |
|  | tweets_p | ssmba | 0.5258 | 0.4666 |
|  | tweets_p | ts | 0.4617 | 0.5275 |
|  | AVG | - | 0.6692 | 0.6570 |


## LLM Fine-tuning

| model | dataset | da_method | macro_precision | macro_recall |
| --- | --- | --- | --- | --- |
| codegen | app | aeda | 0.3798 | 0.3563 |
|  | app | c2l | 0.3727 | 0.3375 |
|  | app | none | 0.3906 | 0.3812 |
|  | app | ssmba | 0.3664 | 0.3187 |
|  | app | ts | 0.3607 | 0.3000 |
|  | gerrit | aeda | 0.3955 | 0.3504 |
|  | gerrit | c2l | 0.5120 | 0.5059 |
|  | gerrit | none | 0.4147 | 0.3369 |
|  | gerrit | ssmba | 0.4910 | 0.4417 |
|  | gerrit | ts | 0.4361 | 0.3726 |
|  | github | aeda | 0.4211 | 0.0397 |
|  | github | c2l | 0.3933 | 0.3272 |
|  | github | none | 0.3727 | 0.3727 |
|  | github | ssmba | 0.5605 | 0.3843 |
|  | github | ts | 0.7564 | 0.2883 |
|  | jira | aeda | 0.5930 | 0.6213 |
|  | jira | c2l | 0.6049 | 0.5776 |
|  | jira | none | 0.4601 | 0.5065 |
|  | jira | ssmba | 0.4469 | 0.5357 |
|  | jira | ts | 0.4604 | 0.5036 |
|  | so | aeda | 0.6546 | 0.6597 |
|  | so | c2l | 0.6484 | 0.6511 |
|  | so | none | 0.5419 | 0.4874 |
|  | so | ssmba | 0.6219 | 0.6282 |
|  | so | ts | 0.5845 | 0.5792 |
|  | tweets | aeda | 0.3173 | 0.9839 |
|  | tweets | c2l | 0.3311 | 0.9153 |
|  | tweets | none | 0.3208 | 0.9274 |
|  | tweets | ssmba | 0.3056 | 0.9765 |
|  | tweets | ts | 0.2984 | 0.8843 |
|  | tweets_n | aeda | 0.3223 | 0.7270 |
|  | tweets_n | c2l | 0.3190 | 0.8340 |
|  | tweets_n | none | 0.3193 | 0.5406 |
|  | tweets_n | ssmba | 0.3258 | 0.8333 |
|  | tweets_n | ts | 0.3314 | 0.9762 |
|  | tweets_p | aeda | 0.3178 | 0.8696 |
|  | tweets_p | c2l | 0.3134 | 0.9457 |
|  | tweets_p | none | 0.2083 | 0.2939 |
|  | tweets_p | ssmba | 0.3209 | 0.9457 |
|  | tweets_p | ts | 0.3432 | 0.8771 |
|  | AVG | - | 0.4234 | 0.5849 |
| deepseek | app | aeda | 0.4539 | 0.4257 |
|  | app | c2l | 0.5037 | 0.5537 |
|  | app | none | 0.4674 | 0.4537 |
|  | app | ssmba | 0.5304 | 0.5785 |
|  | app | ts | 0.7027 | 0.5877 |
|  | gerrit | aeda | 0.2550 | 0.3333 |
|  | gerrit | c2l | 0.2550 | 0.3333 |
|  | gerrit | none | 0.3656 | 0.3304 |
|  | gerrit | ssmba | 0.2550 | 0.3333 |
|  | gerrit | ts | 0.4386 | 0.3504 |
|  | github | aeda | 0.5288 | 0.0156 |
|  | github | c2l | 0.3164 | 0.2641 |
|  | github | none | 0.2500 | 0.0056 |
|  | github | ssmba | 0.4193 | 0.2175 |
|  | github | ts | 0.5043 | 0.3335 |
|  | jira | aeda | 0.2167 | 0.1957 |
|  | jira | c2l | 0.2753 | 0.2184 |
|  | jira | none | 0.3451 | 0.3674 |
|  | jira | ssmba | 0.3257 | 0.3467 |
|  | jira | ts | 0.3972 | 0.4248 |
|  | so | aeda | 0.2911 | 0.3241 |
|  | so | c2l | 0.3449 | 0.3348 |
|  | so | none | 0.5388 | 0.5433 |
|  | so | ssmba | 0.2880 | 0.3655 |
|  | so | ts | 0.3036 | 0.3209 |
|  | tweets | aeda | 0.3710 | 0.3732 |
|  | tweets | c2l | 0.3208 | 0.5994 |
|  | tweets | none | 0.3315 | 0.5811 |
|  | tweets | ssmba | 0.3206 | 0.7835 |
|  | tweets | ts | 0.2970 | 0.5747 |
|  | tweets_n | aeda | 0.3366 | 0.8929 |
|  | tweets_n | c2l | 0.3907 | 0.4175 |
|  | tweets_n | none | 0.3808 | 0.4979 |
|  | tweets_n | ssmba | 0.3224 | 0.8123 |
|  | tweets_n | ts | 0.3308 | 0.7328 |
|  | tweets_p | aeda | 0.2934 | 0.3814 |
|  | tweets_p | c2l | 0.3339 | 0.6966 |
|  | tweets_p | none | 0.2931 | 0.2532 |
|  | tweets_p | ssmba | 0.2613 | 0.6004 |
|  | tweets_p | ts | 0.4726 | 0.4660 |
|  | AVG | - | 0.3657 | 0.4305 |
| phi | app | aeda | 0.7193 | 0.6002 |
|  | app | c2l | 0.4531 | 0.4720 |
|  | app | none | 0.4575 | 0.4660 |
|  | app | ssmba | 0.4666 | 0.4627 |
|  | app | ts | 0.4609 | 0.4688 |
|  | gerrit | aeda | 0.5551 | 0.4244 |
|  | gerrit | c2l | 0.5692 | 0.3788 |
|  | gerrit | none | 0.2591 | 0.2240 |
|  | gerrit | ssmba | 0.2534 | 0.3033 |
|  | gerrit | ts | 0.5581 | 0.4567 |
|  | github | aeda | 0.4306 | 0.1629 |
|  | github | c2l | 0.3972 | 0.3662 |
|  | github | none | 0.1882 | 0.0782 |
|  | github | ssmba | 0.6556 | 0.3188 |
|  | github | ts | 0.4587 | 0.4238 |
|  | jira | aeda | 0.6318 | 0.5391 |
|  | jira | c2l | 0.6421 | 0.5646 |
|  | jira | none | 0.6341 | 0.3080 |
|  | jira | ssmba | 0.6204 | 0.5928 |
|  | jira | ts | 0.6335 | 0.5598 |
|  | so | aeda | 0.6564 | 0.6287 |
|  | so | c2l | 0.6605 | 0.6401 |
|  | so | none | 0.6031 | 0.5102 |
|  | so | ssmba | 0.6532 | 0.6337 |
|  | so | ts | 0.6460 | 0.6432 |
|  | tweets | aeda | 0.2937 | 0.7484 |
|  | tweets | c2l | 0.3244 | 0.9486 |
|  | tweets | none | 0.4379 | 0.5044 |
|  | tweets | ssmba | 0.4062 | 0.9048 |
|  | tweets | ts | 0.4283 | 0.7620 |
|  | tweets_n | aeda | 0.3710 | 0.6802 |
|  | tweets_n | c2l | 0.5380 | 0.7627 |
|  | tweets_n | none | 0.5120 | 0.4452 |
|  | tweets_n | ssmba | 0.2611 | 0.5171 |
|  | tweets_n | ts | 0.2508 | 0.5160 |
|  | tweets_p | aeda | 0.2819 | 0.6139 |
|  | tweets_p | c2l | 0.3943 | 0.7670 |
|  | tweets_p | none | 0.3838 | 0.6123 |
|  | tweets_p | ssmba | 0.4290 | 0.6443 |
|  | tweets_p | ts | 0.3840 | 0.7736 |
|  | AVG | - | 0.4740 | 0.5357 |


## LLM Prompting

| model | dataset | da_method | k | macro_precision | macro_recall |
| --- | --- | --- | --- | --- | --- |
| gpt-4.1-nano_report | app | aeda | k1 | 0.5461 | 0.5509 |
|  | app | aeda | k2 | 0.4602 | 0.4382 |
|  | app | aeda | k4 | 0.5173 | 0.5414 |
|  | app | c2l | k2 | 0.4702 | 0.4350 |
|  | app | c2l | k4 | 0.4667 | 0.4475 |
|  | app | none | k1 | 0.6023 | 0.5880 |
|  | app | none | k2 | 0.6023 | 0.5787 |
|  | app | none | k4 | 0.4594 | 0.4350 |
|  | app | ssmba | k1 | 0.5333 | 0.5231 |
|  | app | ssmba | k2 | 0.4565 | 0.4042 |
|  | app | ssmba | k4 | 0.5387 | 0.5662 |
|  | app | ts | k1 | 0.5216 | 0.5539 |
|  | app | ts | k2 | 0.5595 | 0.5662 |
|  | app | ts | k4 | 0.4596 | 0.4319 |
|  | gerrit | aeda | k1 | 0.4439 | 0.2440 |
|  | gerrit | aeda | k2 | 0.5690 | 0.2120 |
|  | gerrit | aeda | k4 | 0.4464 | 0.2971 |
|  | gerrit | c2l | k2 | 0.5398 | 0.2616 |
|  | gerrit | c2l | k4 | 0.5125 | 0.2944 |
|  | gerrit | none | k1 | 0.4826 | 0.3037 |
|  | gerrit | none | k2 | 0.4472 | 0.2826 |
|  | gerrit | none | k4 | 0.4358 | 0.1856 |
|  | gerrit | ssmba | k1 | 0.3542 | 0.1536 |
|  | gerrit | ssmba | k2 | 0.5682 | 0.1522 |
|  | gerrit | ssmba | k4 | 0.5537 | 0.2947 |
|  | gerrit | ts | k1 | 0.5046 | 0.2788 |
|  | gerrit | ts | k2 | 0.5225 | 0.2430 |
|  | gerrit | ts | k4 | 0.5456 | 0.2824 |
|  | github | aeda | k1 | 0.5176 | 0.1315 |
|  | github | aeda | k2 | 0.4296 | 0.1553 |
|  | github | aeda | k4 | 0.4146 | 0.2649 |
|  | github | c2l | k2 | 0.4538 | 0.1495 |
|  | github | c2l | k4 | 0.3844 | 0.2087 |
|  | github | none | k1 | 0.4560 | 0.1637 |
|  | github | none | k2 | 0.4382 | 0.2184 |
|  | github | none | k4 | 0.3736 | 0.2540 |
|  | github | ssmba | k1 | 0.5656 | 0.0843 |
|  | github | ssmba | k2 | 0.4591 | 0.1453 |
|  | github | ssmba | k4 | 0.3897 | 0.2180 |
|  | github | ts | k1 | 0.4190 | 0.1211 |
|  | github | ts | k2 | 0.4650 | 0.1554 |
|  | github | ts | k4 | 0.3881 | 0.1965 |
|  | jira | aeda | k1 | 0.4738 | 0.3487 |
|  | jira | aeda | k2 | 0.4369 | 0.3563 |
|  | jira | aeda | k4 | 0.4660 | 0.4006 |
|  | jira | c2l | k2 | 0.4616 | 0.3444 |
|  | jira | c2l | k4 | 0.4679 | 0.3941 |
|  | jira | none | k1 | 0.4630 | 0.3504 |
|  | jira | none | k2 | 0.4454 | 0.4081 |
|  | jira | none | k4 | 0.4530 | 0.4108 |
|  | jira | ssmba | k1 | 0.4641 | 0.3357 |
|  | jira | ssmba | k2 | 0.4601 | 0.3514 |
|  | jira | ssmba | k4 | 0.5108 | 0.3912 |
|  | jira | ts | k1 | 0.4600 | 0.3490 |
|  | jira | ts | k2 | 0.4484 | 0.3722 |
|  | jira | ts | k4 | 0.4722 | 0.3942 |
|  | so | aeda | k1 | 0.5407 | 0.4082 |
|  | so | aeda | k2 | 0.5321 | 0.2985 |
|  | so | aeda | k4 | 0.5263 | 0.3763 |
|  | so | c2l | k2 | 0.5312 | 0.3026 |
|  | so | c2l | k4 | 0.5422 | 0.3970 |
|  | so | none | k1 | 0.5632 | 0.3835 |
|  | so | none | k2 | 0.5632 | 0.3908 |
|  | so | none | k4 | 0.5408 | 0.3954 |
|  | so | ssmba | k1 | 0.5371 | 0.3232 |
|  | so | ssmba | k2 | 0.5086 | 0.2529 |
|  | so | ssmba | k4 | 0.5426 | 0.3726 |
|  | so | ts | k1 | 0.5415 | 0.3494 |
|  | so | ts | k2 | 0.5395 | 0.2879 |
|  | so | ts | k4 | 0.5500 | 0.3650 |
|  | tweets | aeda | k1 | 0.5839 | 0.5327 |
|  | tweets | aeda | k2 | 0.5535 | 0.4620 |
|  | tweets | aeda | k4 | 0.5579 | 0.5108 |
|  | tweets | c2l | k2 | 0.5801 | 0.3660 |
|  | tweets | c2l | k4 | 0.5356 | 0.5090 |
|  | tweets | ssmba | k1 | 0.5801 | 0.4955 |
|  | tweets | ssmba | k2 | 0.6023 | 0.4895 |
|  | tweets | ssmba | k4 | 0.5668 | 0.4632 |
|  | tweets | ts | k1 | 0.5728 | 0.5077 |
|  | tweets | ts | k2 | 0.5495 | 0.4569 |
|  | tweets | ts | k4 | 0.5575 | 0.5372 |
|  | tweets_n | aeda | k1 | 0.5220 | 0.4142 |
|  | tweets_n | aeda | k2 | 0.4643 | 0.3614 |
|  | tweets_n | aeda | k4 | 0.4794 | 0.3842 |
|  | tweets_n | c2l | k2 | 0.4932 | 0.3877 |
|  | tweets_n | c2l | k4 | 0.5215 | 0.3560 |
|  | tweets_n | none | k1 | 0.4868 | 0.4881 |
|  | tweets_n | none | k2 | 0.4632 | 0.4221 |
|  | tweets_n | none | k4 | 0.6081 | 0.3724 |
|  | tweets_n | one | k1 | 0.5978 | 0.5234 |
|  | tweets_n | one | k2 | 0.5923 | 0.5028 |
|  | tweets_n | one | k4 | 0.5645 | 0.6118 |
|  | tweets_n | ssmba | k1 | 0.7588 | 0.4744 |
|  | tweets_n | ssmba | k2 | 0.4640 | 0.3285 |
|  | tweets_n | ssmba | k4 | 0.5602 | 0.3377 |
|  | tweets_n | ts | k1 | 0.6977 | 0.5443 |
|  | tweets_n | ts | k2 | 0.4760 | 0.5026 |
|  | tweets_n | ts | k4 | 0.7190 | 0.3960 |
|  | tweets_p | aeda | k1 | 0.8491 | 0.3929 |
|  | tweets_p | aeda | k2 | 0.7609 | 0.3803 |
|  | tweets_p | aeda | k4 | 0.6630 | 0.4387 |
|  | tweets_p | c2l | k2 | 0.8358 | 0.3202 |
|  | tweets_p | c2l | k4 | 0.7374 | 0.4206 |
|  | tweets_p | none | k1 | 0.5880 | 0.4801 |
|  | tweets_p | none | k2 | 0.6593 | 0.4956 |
|  | tweets_p | none | k4 | 0.7312 | 0.5292 |
|  | tweets_p | ssmba | k1 | 0.5371 | 0.3150 |
|  | tweets_p | ssmba | k2 | 0.5913 | 0.3203 |
|  | tweets_p | ssmba | k4 | 0.7225 | 0.3729 |
|  | tweets_p | ts | k1 | 0.6621 | 0.4019 |
|  | tweets_p | ts | k2 | 0.7100 | 0.4522 |
|  | tweets_p | ts | k4 | 0.7559 | 0.5108 |
|  | AVG | - | nan | 0.5342 | 0.3722 |
| gpt-5-nano_report | app | aeda | k1 | 0.4689 | 0.4722 |
|  | app | aeda | k2 | 0.4754 | 0.4815 |
|  | app | aeda | k4 | 0.4633 | 0.4722 |
|  | app | c2l | k2 | 0.4626 | 0.4630 |
|  | app | c2l | k4 | 0.7226 | 0.5972 |
|  | app | none | k1 | 0.4669 | 0.4630 |
|  | app | none | k2 | 0.5971 | 0.5880 |
|  | app | none | k4 | 0.7226 | 0.5972 |
|  | app | ssmba | k1 | 0.4626 | 0.4630 |
|  | app | ssmba | k2 | 0.4717 | 0.4567 |
|  | app | ssmba | k4 | 0.4673 | 0.4722 |
|  | app | ts | k1 | 0.6031 | 0.5972 |
|  | app | ts | k2 | 0.6031 | 0.5972 |
|  | app | ts | k4 | 0.4673 | 0.4722 |
|  | gerrit | aeda | k1 | 0.5779 | 0.4257 |
|  | gerrit | aeda | k2 | 0.5833 | 0.4773 |
|  | gerrit | aeda | k4 | 0.5361 | 0.4438 |
|  | gerrit | c2l | k2 | 0.5759 | 0.4638 |
|  | gerrit | c2l | k4 | 0.5694 | 0.4507 |
|  | gerrit | none | k1 | 0.5777 | 0.4803 |
|  | gerrit | none | k2 | 0.5791 | 0.5110 |
|  | gerrit | none | k4 | 0.5687 | 0.4773 |
|  | gerrit | ssmba | k1 | 0.6018 | 0.4079 |
|  | gerrit | ssmba | k2 | 0.6051 | 0.4241 |
|  | gerrit | ssmba | k4 | 0.5553 | 0.4759 |
|  | gerrit | ts | k1 | 0.5790 | 0.4520 |
|  | gerrit | ts | k2 | 0.5762 | 0.4800 |
|  | gerrit | ts | k4 | 0.5568 | 0.4274 |
|  | github | aeda | k1 | 0.4971 | 0.4739 |
|  | github | aeda | k2 | 0.5023 | 0.4620 |
|  | github | aeda | k4 | 0.5443 | 0.4574 |
|  | github | c2l | k2 | 0.5117 | 0.4569 |
|  | github | c2l | k4 | 0.5159 | 0.4285 |
|  | github | none | k1 | 0.5070 | 0.4599 |
|  | github | none | k2 | 0.4813 | 0.4533 |
|  | github | none | k4 | 0.4953 | 0.4520 |
|  | github | ssmba | k1 | 0.5017 | 0.4507 |
|  | github | ssmba | k2 | 0.4898 | 0.4525 |
|  | github | ssmba | k4 | 0.5203 | 0.4414 |
|  | github | ts | k1 | 0.4831 | 0.4498 |
|  | github | ts | k2 | 0.5061 | 0.4481 |
|  | github | ts | k4 | 0.5061 | 0.4226 |
|  | jira | aeda | k1 | 0.5231 | 0.5604 |
|  | jira | aeda | k2 | 0.5173 | 0.5507 |
|  | jira | aeda | k4 | 0.5226 | 0.5323 |
|  | jira | c2l | k2 | 0.5127 | 0.5510 |
|  | jira | c2l | k4 | 0.5216 | 0.5424 |
|  | jira | none | k1 | 0.5020 | 0.5388 |
|  | jira | none | k2 | 0.4946 | 0.5430 |
|  | jira | none | k4 | 0.4963 | 0.5276 |
|  | jira | ssmba | k1 | 0.5254 | 0.5674 |
|  | jira | ssmba | k2 | 0.5228 | 0.5521 |
|  | jira | ssmba | k4 | 0.5328 | 0.5476 |
|  | jira | ts | k1 | 0.5182 | 0.5510 |
|  | jira | ts | k2 | 0.5089 | 0.5427 |
|  | jira | ts | k4 | 0.5224 | 0.5298 |
|  | so | aeda | k1 | 0.6092 | 0.5907 |
|  | so | aeda | k2 | 0.6081 | 0.5946 |
|  | so | aeda | k4 | 0.6072 | 0.5743 |
|  | so | c2l | k2 | 0.6051 | 0.5800 |
|  | so | c2l | k4 | 0.5976 | 0.5629 |
|  | so | none | k1 | 0.6066 | 0.5942 |
|  | so | none | k2 | 0.6090 | 0.5981 |
|  | so | none | k4 | 0.6040 | 0.5865 |
|  | so | ssmba | k1 | 0.6037 | 0.5782 |
|  | so | ssmba | k2 | 0.6049 | 0.5838 |
|  | so | ssmba | k4 | 0.6088 | 0.5637 |
|  | so | ts | k1 | 0.6052 | 0.5839 |
|  | so | ts | k2 | 0.6115 | 0.5840 |
|  | so | ts | k4 | 0.5994 | 0.5640 |
|  | tweets | aeda | k1 | 0.8206 | 0.7039 |
|  | tweets | aeda | k2 | 0.8412 | 0.6734 |
|  | tweets | aeda | k4 | 0.7913 | 0.6693 |
|  | tweets | c2l | k2 | 0.8419 | 0.6894 |
|  | tweets | c2l | k4 | 0.8304 | 0.6659 |
|  | tweets | ssmba | k1 | 0.7519 | 0.6593 |
|  | tweets | ssmba | k2 | 0.8187 | 0.6753 |
|  | tweets | ssmba | k4 | 0.8420 | 0.6927 |
|  | tweets | ts | k1 | 0.7956 | 0.6699 |
|  | tweets | ts | k2 | 0.8117 | 0.7018 |
|  | tweets | ts | k4 | 0.8187 | 0.6805 |
|  | tweets_n | aeda | k1 | 0.5775 | 0.6234 |
|  | tweets_n | aeda | k2 | 0.5634 | 0.4960 |
|  | tweets_n | aeda | k4 | 0.5940 | 0.4870 |
|  | tweets_n | c2l | k2 | 0.5557 | 0.5176 |
|  | tweets_n | c2l | k4 | 0.5963 | 0.4939 |
|  | tweets_n | none | k1 | 0.5528 | 0.6762 |
|  | tweets_n | none | k2 | 0.6269 | 0.6412 |
|  | tweets_n | none | k4 | 0.5755 | 0.5394 |
|  | tweets_n | one | k1 | 0.8411 | 0.6962 |
|  | tweets_n | one | k2 | 0.8289 | 0.6915 |
|  | tweets_n | one | k4 | 0.8326 | 0.7092 |
|  | tweets_n | ssmba | k1 | 0.6032 | 0.6570 |
|  | tweets_n | ssmba | k2 | 0.5785 | 0.5142 |
|  | tweets_n | ssmba | k4 | 0.6431 | 0.4945 |
|  | tweets_n | ts | k1 | 0.5930 | 0.6718 |
|  | tweets_n | ts | k2 | 0.5937 | 0.5077 |
|  | tweets_n | ts | k4 | 0.5805 | 0.4816 |
|  | tweets_p | aeda | k1 | 0.6158 | 0.6295 |
|  | tweets_p | aeda | k2 | 0.6378 | 0.6242 |
|  | tweets_p | aeda | k4 | 0.5808 | 0.5833 |
|  | tweets_p | c2l | k2 | 0.5648 | 0.6008 |
|  | tweets_p | c2l | k4 | 0.6258 | 0.6638 |
|  | tweets_p | none | k1 | 0.5873 | 0.5895 |
|  | tweets_p | none | k2 | 0.6428 | 0.6445 |
|  | tweets_p | none | k4 | 0.6296 | 0.6322 |
|  | tweets_p | ssmba | k1 | 0.6024 | 0.6061 |
|  | tweets_p | ssmba | k2 | 0.6400 | 0.6568 |
|  | tweets_p | ssmba | k4 | 0.6029 | 0.6035 |
|  | tweets_p | ts | k1 | 0.5908 | 0.6662 |
|  | tweets_p | ts | k2 | 0.6401 | 0.6587 |
|  | tweets_p | ts | k4 | 0.6054 | 0.6243 |
|  | AVG | - | nan | 0.5940 | 0.5521 |
