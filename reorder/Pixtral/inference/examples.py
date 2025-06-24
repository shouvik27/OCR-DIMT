from ..fine_tune import *

def get_few_shot_examples():
    # Example 3
    example3_img_path = r"E:\\Data\\output_train_img\\doc_00c51cec52464b73f2a850591b7e5457fc5d2de1.page_2.png"
    example3_human = {
        "content": [
            {"type": "image_url", "image_url": {"url": image_file_to_base64(example3_img_path)}}
        ]
    }
    example3_ai = {
        "content": (
            "Ordered Text output: NAME OF PROJECT: ______________________________________________________ PROJECT DESCRIPTION "
            "Most project team members are so close to a project that they have difficulty describing their project in simple terms. "
            "It is important to be able to describe your project in three-to-four sentences that will convey the opportunity and the benefit. "
            "Your project description should include the: • Name of your project. • The problem it is trying to solve. • The benefit of solving the problem (save money, save time, more effective, etc.). "
            "• The specific audience whose lives will improve because of it • How the audience(s) endeavors will improve. • The hoped-for benefit to UC Berkeley if the project is successful. "
            "The description should have the following structure. (Name of Project) will enable UC Berkeley to (benefit) by (problem it will solve). "
            "When successfully completed, (audience) will become more effective by (anticipated improvement). YOUR PROJECT DESCRIPTION: "
            "AUDIENCES Below is a list of campus groups to consider when developing your communication plan."
        )
    }
    # Example 6
    example6_img_path = r"E:\\Data\\output_train_img\\doc_0a1ea1410d79980d9a21d3e0bd16d1c920deccff.page_15.png"
    example6_human = {
        "content": [
            {"type": "image_url", "image_url": {"url": image_file_to_base64(example6_img_path)}}
        ]
    }
    example6_ai = {
        "content": (
            "Ordered Text output: deems it necessary, use the national research facilities to conduct research. 5A. Research institutions (1) "
            "The Minister may, after consultation with the Foundation, declare by notice in the Gazette an institution conducting research, as a research institution and eligible to receive grant funding from the Foundation. "
        )
    }
    # Example 9
    example9_img_path = r"E:\\Data\\output_train_img\\doc_00aff3757f87ebdfa666e19cda073e0d3a30943c.page_2.png"
    example9_human = {
        "content": [
            {"type": "image_url", "image_url": {"url": image_file_to_base64(example9_img_path)}}
        ]
    }
    example9_ai = {
        "content": (
            "Ordered Text output: The Class of 2019, 2020, 2021 and 2022 fell below in both GRE scores in relation to national averages. "
            "In identified is that we struggle to meet the benchmark in relation to GRE scores, though we await most recent. The GPAs have a higher weighting than GRE in our admissions evaluation process. "
            "This can explain why GPAs but not the GREs. As class size has increased over the past 4 years, GRE scores have remained relatively CASPA benchmark averages. "
            "The admitted students tend to perform very well in our PA Program, consistently accordance with ARC-PA Standards and find employment. "
            "Presently, we do not intend to change our admissions applicants with higher GRE scores as our program becomes more well-known for producing highly qualified candidates. "
            "Overall, the HPU PA Program is partially meeting the program established goal of recruiting highly Outcome Measure B: Matriculated student prior healthcare experience. "
            "Benchmark: Matriculated student mean healthcare experience will be at least 1,000 hours and greater than average Average level of patient care/responsibility is 2 on a 1-4-point scale. "
            "Data: Class of 2019, Class of 2020, Class of 2021, Class of 2022, Class of 2023; Healthcare Experience – Average # of Hours: 3,370, 1,835, 3,231, 3082, 2803; "
            "Healthcare Experience – Median # of Hours: 2,153, 1,314, 1,990, 2269, 1760; Healthcare Responsibility Level: 2.09, 1.94, 2.11, 2.04, 2.10; Analysis: • HPU Class of 2019 "
            "attained both benchmarks related to healthcare experience. • HPU Class of 2020 attained the first benchmark (total hours) but fell just below the benchmark in average. "
            "• HPU Class of 2021 attained both benchmarks related to healthcare experience. • HPU Class of pf 2022 attained both benchmarks related to healthcare experience. "
            "• HPU Class of pf 2023 attained both benchmarks related to healthcare experience. Over the last five admission years, there has been a fluctuation of average and median "
            "healthcare experience responsibility. This is in contrast to the total number of applications to the PA program, which has increased."
        )
    }
    # Example 10
    example10_img_path = r"E:\\Data\\output_train_img\\doc_00b5005717fb8925b94ecc069b9ad4954590f125.page_14.png"
    example10_human = {
        "content": [
            {"type": "image_url", "image_url": {"url": image_file_to_base64(example10_img_path)}}
        ]
    }
    example10_ai = {
        "content": (
            "Ordered Text output: Jose Rodriguez West Coast Burrito © Southern Foodways Alliance | www.southernfoodways.org 15 [0:10:11.7] Delerme: Okay, six. "
            "Wow. [0:10:12.9] Rodriguez: And all the customers, they come, so they're welcome to try all the salsas they want with chips also. "
            "So we have chips and salsa bar so they can—we have avocado salsa—I mean guacamole, guacamole salsa. We have green tomatillo, red tomatillo. "
            "We have a spicy, we have chipotle. We have like six different kinds of salsas on the salsa bar, and people like it. The people like the salsa bar. "
            "The only thing that I'm trying to fix a little bit, because they've been making the salsas a little hot, so we are going to make them mild, "
            "just a little bit of spice, but not a lot, so that way, everybody can enjoy the salsa, you know. [0:10:53.4] Delerme: Okay. Okay, that's fantastic. "
            "And how about your family? Does your family get involved here or is this just you that's involved in the business? [0:11:00.9] Rodriguez: No. "
            "Actually, just my wife and me, so that's the only— [0:11:10.0] Delerme: Oh, so she is too?"
        )
    }
    # Return as a list of alternating human/AI messages
    return [example3_human, example3_ai, example6_human, example6_ai, example9_human, example9_ai, example10_human, example10_ai]
