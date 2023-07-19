from transformers import pipeline
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", truncation=True)
text = '''On the evening of November 12, speaking at the Asia-Pacific Economic Cooperation (APEC) Summit with the 
participation of Russian President Vladimir Putin, Chinese President Xi Jinping and US President Joe Biden, 
President Nguyen Xuan Phuc proposed to APEC, at this special moment, members need to overcome differences to think 
together and act for the benefit of themselves and the whole community, first of all. sustainable recovery from the 
pandemic and inclusive growth. The 2021 APEC meetings will be conducted entirely online as host New Zealand is still 
closing its borders to prevent COVID-19. Opening remarks At the world's last multilateral conference this year, 
Prime Minister Jacinda Ardern said that each economy in APEC has experienced the COVID-19 pandemic differently and 
responded to the pandemic in its own ways, but all face it. with the same basic issues, including promoting 
vaccination, maintaining production, business and jobs for people, ensuring safe travel between countries, 
accompanied by a strong economic recovery and inclusivity. Together expressed APEC's determination as never before to 
overcome the pandemic, accelerate economic recovery, go hand in hand with climate change response and promote 
inclusive growth for all. President Nguyen Xuan Phuc emphasized that the world's difficult and painful battle with 
the COVID-19 pandemic over the past two years has forced economies to reflect and rethink many regional issues. and 
global governance, especially the vulnerability and lack of preparedness to epidemics and climate change, as well as 
the inadequacies and limitations of the global governance system in crisis management along with the inequality in 
and between economies. President Nguyen Xuan Phuc emphasized that history shows that each time it overcomes a crisis, 
APEC proves its strong vitality and cohesive role. In today's difficulties, more than ever, APEC - which contributes 
more than 60% of GDP and nearly half of global trade - needs to continue promoting its role as a driving force for 
global economic growth, along with the affirmation. role as the center of initiating innovative ideas and new 
development trends. At the same time, APEC needs to proactively expand economic linkages in recovery and sustainable 
growth, lead the shaping of the world economy after the pandemic, and contribute to strengthening effective global 
economic governance. In the joint statement of the Conference and the Action Plan to implement the APEC Vision 2040, 
approved by President Nguyen Xuan Phuc and other leaders, affirms the determination to use all economic tools. macro 
available to address the adverse consequences of the COVID-19 pandemic, sustain the economic recovery, 
while maintaining long-term fiscal sustainability. Leaders are committed to boosting production and provide COVID-19 
vaccines through technology transfer and removal of export restrictions on medical devices. Along with increased 
cooperation in COVID-19 testing and passport vaccines when reopening borders and when people's movement between 
economies increases. The leaders also pledged to stop increasing subsidies for fossil fuel extraction and use, 
creating a basis for discussion on issues related to climate change in future APEC meetings. President Nguyen Xuan 
Phuc and other leaders also agreed that Thailand will hold the role of APEC Chair in 2022. New Zealand Prime Minister 
Jacindar Ardern has transferred this responsibility to Prime Minister Prayut Chan-ocha whose symbol is the roof. 
rowing of the Maori people, so that Thailand can bring the APEC boat to harmonious cooperation and change in the 
present towards a common future. President Nguyen Xuan Phuc affirmed that Vietnam will cooperate with Thailand and 
other members. successfully organized the APEC Year 2022 with 3 priorities, namely Open to all opportunities, 
Connected in all aspects and Balanced in all aspects.'''
print(summarizer(text))
