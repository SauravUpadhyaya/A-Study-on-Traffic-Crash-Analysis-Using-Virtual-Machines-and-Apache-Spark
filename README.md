# A-Study-on-Traffic-Crash-Analysis-Using-Virtual-Machines-and-Apache-Spark

Chicago is considered one of the congested cities in the United States, with a population of around 3 million (2024). Every year, it is estimated that 100,000 vehicles are involved in accidents. Thus, it is necessary to analyze the crash data to understand the crash trends and patterns, aiming to provide suggestions and recommendations to the Department of Public Safety authorities. In this project, crash-relevant datasets (~ 2GB), such as crashes, people, vehicles, and Vision Zero traffic fatalities owned by the Chicago Police Department (Data Owner), are used to create geo-spatial visualizations.

In this project, I created some major geospatial visualizations obtained from data analysis using PySpark in Chicago Traffic Crashes (crashes, people, vehicles, and zero vision fatalities) dataset. In my application, I have used the “apache-spark” image (link) for managing one master and two workers. This is done to simplify dependency management and ensure a consistent runtime environment across all nodes. 1 Spark Master and 2 Spark Workers are the core components in my Docker containers.


<img width="629" height="556" alt="Screenshot 2025-10-18 at 3 43 43 PM" src="https://github.com/user-attachments/assets/c5f674d4-5143-4ed7-a8fa-d8476caaefb6" />

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Example 1: [Total number of crashes happening in each city] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/crash_markers.html)

<img width="628" height="568" alt="Screenshot 2025-10-18 at 3 38 58 PM" src="https://github.com/user-attachments/assets/8494b15d-afa3-4884-8875-dcdc86fa93fc" />


In the above figure, I see the number of crashes happening per city in Chicago.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Example 2: [Fatal Crashes: Cyclists] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/people_fatal_ped_cyc_map.html)

<img width="463" height="312" alt="Screenshot 2025-10-08 at 9 28 51 PM" src="https://github.com/user-attachments/assets/54da7e06-0805-4cca-b5f6-c465edd8db9f" />

                               
In the above figure,  I see the number of blue markers representing the death of the Cyclists.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Example 3: [Fatal Crashes: Pedestrians] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/people_fatal_ped_cyc_map.html)

<img width="351" height="325" alt="image" src="https://github.com/user-attachments/assets/fe43675b-a7ed-403f-8434-9b9b5d905b1d" />


In the above figure, I see the red clusters representing the different cities in Chicago.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Example 4: [Fatal Traffic Accidents Analysis] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/fatal_policy_map.html)

<img width="614" height="346" alt="Screenshot 2025-10-08 at 8 43 05 PM" src="https://github.com/user-attachments/assets/0100c9e4-b008-4095-9f56-ab72e5174093" />

In the above figure, I see that the fatal crashes are clustered in certain areas in which age groups and road conditions are key factors.




Some other geospatial visualizations obtained from data analysis using PySpark:

[Crashes (top 5 causes)] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/crash_top5_causes_map.html)

[Visualizing Crashes, Injury, and Fatal] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/crash_admin_map.html)

[Crashes (Heatmap)] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/crash_heatmap.html)

[Risk analysis] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/risk_map.html)

[Vehicles' fatal heavy hit run visualization] (https://sauravupadhyaya.github.io/traffic_crashes_visualization/vehicles_fatal_heavy_hitrun_map.html)
