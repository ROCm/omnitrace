# from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
from seleniumwire.utils import decode
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException


class BasePage(object):
    def __init__(self, driver):
        self.driver = driver


class MainPage(BasePage):
    def get_titles(self):
        """Verifies that graph_all ID appears in page"""
        graph_all_elements = WebDriverWait(self.driver, 60).until(
            EC.visibility_of_element_located((By.ID, "graph_all"))
        )

        h4_tags_ = graph_all_elements.find_elements("tag name", "h4")
        h4_tags = [elem.text for elem in h4_tags_]

        return h4_tags

    def get_histogram_data(self):
        WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "graph_all"))
        )

        responses = []
        for request in self.driver.requests:
            if request.response:
                if "_dash-update-component" in request.url:
                    responses.append(
                        json.loads(
                            decode(
                                request.response.body,
                                request.response.headers.get(
                                    "Content-Encoding", "identity"
                                ),
                            ).decode("utf-8")
                        )
                    )
        histogram = responses[0]["response"]["graph_select"]["children"][1]["props"][
            "figure"
        ]["data"][0]
        histogram_x = histogram["x"]
        histogram_y = histogram["y"]

        consolidated_hist = dict(zip(sorted(set(histogram_x)), [0] * len(histogram_x)))
        for idx in range(0, len(histogram_x)):
            consolidated_hist[histogram_x[idx]] = (
                consolidated_hist[histogram_x[idx]] + histogram_y[idx]
            )

        return consolidated_hist

    def get_alphabetical_titles(self):
        """Change to alphabetical ordering"""

        element = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "Sort by-filt"))
        )
        element.click()

        element2 = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, ("//*[contains(text(), 'Alphabetical')]"))
            )
        )
        element2.click()

        graph_all_elements = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "graph_all"))
        )

        h4_tags_ = graph_all_elements.find_elements("tag name", "h4")

        h4_tags = [elem.text for elem in h4_tags_]

        return h4_tags

    def get_max_speedup_titles(self):
        """Change to alphabetical ordering"""

        element = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "Sort by-filt"))
        )
        element.click()

        element2 = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, ("//*[contains(text(), 'Max Speedup')]"))
            )
        )
        element2.click()

        graph_all_elements = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "graph_all"))
        )

        h4_tags_ = graph_all_elements.find_elements("tag name", "h4")

        h4_tags = [elem.text for elem in h4_tags_]

        return h4_tags

    def get_min_speedup_titles(self):
        """Change to alphabetical ordering"""

        element = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "Sort by-filt"))
        )
        element.click()

        element2 = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, ("//*[contains(text(), 'Min Speedup')]"))
            )
        )
        element2.click()

        graph_all_elements = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "graph_all"))
        )

        h4_tags_ = graph_all_elements.find_elements("tag name", "h4")

        h4_tags = [elem.text for elem in h4_tags_]

        return h4_tags

    def get_impact_titles(self):
        """Change to alphabetical ordering"""

        element = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "Sort by-filt"))
        )
        element.click()

        element2 = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, ("//*[contains(text(), 'Impact')]"))
            )
        )
        element2.click()

        graph_all_elements = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "graph_all"))
        )

        h4_tags_ = graph_all_elements.find_elements("tag name", "h4")

        h4_tags = [elem.text for elem in h4_tags_]

        return h4_tags

    def get_min_points_titles(self):
        element = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, ("//*[contains(@class,'rc-slider-handle')]"))
            )
        )
        slider = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, ("//*[contains(@id,'points-filt')]"))
            )
        )
        WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, ("//*[contains(@id,'experiment_regex')]"))
            )
        )
        points = element.text
        size = slider.size["width"]
        ActionChains(self.driver).drag_and_drop_by_offset(element, size, 0).perform()

        max_points = int(element.text)

        offset = size / int(max_points)

        points = []

        for i in range(0, max_points + 1):
            time.sleep(10)
            graph_all_elements = WebDriverWait(self.driver, 120).until(
                EC.visibility_of_element_located((By.ID, "graph_all"))
            )

            h4_tags_ = graph_all_elements.find_elements("tag name", "h4")
            h4_tags = [elem.text for elem in h4_tags_]
            points.append({"num points": int(element.text), "titles": h4_tags})
            ActionChains(self.driver).drag_and_drop_by_offset(
                element, -1 * offset, 0
            ).perform()

        return points

    def remove_workload(self):
        points = []

        def check_exists_by_xpath(xpath):
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, xpath))
                )
            except NoSuchElementException:
                return False
            except TimeoutException:
                return False
            return True

        while check_exists_by_xpath(
            "//*[contains(@class,'Select-value-icon') and contains(text(), x)]"
        ):
            element = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located(
                    (
                        By.XPATH,
                        (
                            "//*[contains(@class,'Select-value-icon') and contains(text(), x)]"
                        ),
                    )
                )
            )

            graph_all_elements = WebDriverWait(self.driver, 120).until(
                EC.visibility_of_element_located((By.ID, "graph_all"))
            )

            h4_tags_ = graph_all_elements.find_elements("tag name", "h4")
            h4_tags = [elem.text for elem in h4_tags_]
            points.append({"titles": h4_tags})
            element.click()
            time.sleep(10)

        return points

    def get_plot_data(self):
        WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "graph_all"))
        )

        responses = []
        graphs = []
        for request in self.driver.requests:
            if request.response:
                if "_dash-update-component" in request.url:
                    responses.append(
                        json.loads(
                            decode(
                                request.response.body,
                                request.response.headers.get(
                                    "Content-Encoding", "identity"
                                ),
                            ).decode("utf-8")
                        )
                    )
        # histogram = responses[0]["response"]["graph_select"]["children"][1]["props"][
        #     "figure"
        # ]["data"][0]
        plots = responses[0]["response"]["graph_all"]["children"]
        for plot in plots:
            if "Graph" == plot["type"]:
                plot_data = plot["props"]["figure"]["data"]
                for sub_plot in plot_data:
                    graphs.append(sub_plot)

        return graphs
