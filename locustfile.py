from locust import HttpUser, task, between


class MyUser(HttpUser):
    wait_time = between(0.25, 1)

    @task
    def my_task(self):
        endpoint = "/predict"
        file_list = [
            "000000000872.jpg", "000000157756.jpg", "000000169996.jpg", "000000183246.jpg", "000000215723.jpg",
            "000000221754.jpg", "000000242411.jpg", "000000254814.jpg", "000000255917.jpg", "000000287291.jpg",
            "000000292997.jpg", "000000377946.jpg", "000000393226.jpg", "000000396200.jpg",
            "bus.jpg", "horses.jpg",
            "image1.jpg", "image2.jpg", "image3.jpg", "zidane.jpg"
        ]
        for filename in file_list:
            self.client.post(endpoint, json={"filename": filename})
