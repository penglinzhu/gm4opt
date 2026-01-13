\ Model demo_20260113_104552
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xbf81f5853e61a6a9
Maximize
  1500 area[corn] + 1200 area[wheat] + 1800 area[soybeans]
   + 1600 area[sorghum]
Subject To
 total_area: area[corn] + area[wheat] + area[soybeans] + area[sorghum]
   <= 100
 corn_wheat_ratio: area[corn] - 2 area[wheat] >= 0
 soybeans_sorghum_ratio: area[soybeans] - 0.5 area[sorghum] >= 0
 wheat_sorghum_ratio: area[wheat] - 3 area[sorghum] = 0
Bounds
Generals
 area[corn] area[wheat] area[soybeans] area[sorghum]
End
