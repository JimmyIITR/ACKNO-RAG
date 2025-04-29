import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from dataBase import queries
from langchain_ollama import ChatOllama
import prompts
import articleExtrection
from selectData import tempFileFactText,tempFileFalseFactText,dataPath,llmModel,embeddingModel

from dotenv import load_dotenv

load_dotenv()

FACT_DATA = tempFileFactText()
FALSE_FACT_DATA = tempFileFalseFactText()

DATA_PATH = dataPath()
LLM_MODEL = llmModel()
EMBEDDINGS_MODEL = embeddingModel()

def loadData(dataPath):
    """Load and split documents from specified path"""
    textLoader = TextLoader(file_path=dataPath, autodetect_encoding=True)
    rawDocs = textLoader.load()
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=250, 
        chunk_overlap=24
    )
    return textSplitter.split_documents(documents=rawDocs)

def processLLM(docs):
    """Process documents to create LLM instance and graph documents"""
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    graphTransformer = LLMGraphTransformer(llm=llm)
    graphDocs = graphTransformer.convert_to_graph_documents(docs)
    return llm, graphDocs

def addToGraph(graph, graphDocs):
    """Add processed documents to Neo4j graph"""
    graph.add_graph_documents(
        graphDocs,
        baseEntityLabel=True,
        include_source=True
    )

def getNodesListIDs(graph_docs) -> list[str]:
    """Return a sorted list of all unique node IDs in the given GraphDocuments."""
    return sorted({ node.id for doc in graph_docs for node in doc.nodes })


def extract_counts(paths_str):
    paths = [p for p in paths_str.split("\n") if p.strip()]
    counts = []
    for p in paths:
        parts = p.split(" - ")
        total_nodes = (len(parts) + 1) // 2 # for relation removalas rel will be on odd places
        intermediate = max(0, total_nodes - 2)
        counts.append(intermediate)
    return counts, len(paths)

def dataGeneratorForLogistic(graph, result):
    """ result = {true: [], false: [], combined: []} """
    #below lines are incomplete
    true_nodes = result["true"]
    false_nodes = result["false"]
    combined = result["combined"]

    tt_horiz, tt_vert = [], []
    tf_horiz, tf_vert = [], []

    # 1 True–True
    for i, en1 in enumerate(true_nodes):
        for en2 in true_nodes[i+1:]:
            print(en1, en2)
            paths_str = queries.twoNodeConnection(en1, en2, combined, graph)
            print(paths_str)
            counts, num_paths = extract_counts(paths_str)
            tt_horiz.extend(counts)
            tt_vert.append(num_paths)

    # 2 True–False
    for en1 in true_nodes:
        for en2 in false_nodes:
            print(en1, en2)
            paths_str = queries.twoNodeConnection(en1, en2, combined, graph)
            print(paths_str)
            counts, num_paths = extract_counts(paths_str)
            tf_horiz.extend(counts)
            tf_vert.append(num_paths)

    return {
        "true_true_horizontal":  tt_horiz,
        "true_true_vertical":    tt_vert,
        "true_false_horizontal": tf_horiz,
        "true_false_vertical":   tf_vert,
    }

def handleDataIngestion(index=1):
    print(f"Loading and processing data for {index}")
    trueDocuements = loadData(FACT_DATA)
    # llmModel, graphDocuments = processLLM(trueDocuements)
    # trueNodes = getNodesListIDs(graphDocuments)
    # print(f"FACT data completeted for {index}")
    # print(trueNodes) ## temp
    # print(trueNodes)
    falseDocuments = loadData(FALSE_FACT_DATA)
    # llmModel, graphDocuments = processLLM(falseDocuments)
    # falseNodes = getNodesListIDs(graphDocuments)
    # print(f"FALSE data completeted for {index}")
    # print(falseNodes) ## temp
    # print(falseNodes)

    #new combiend data graph generation
    combinedData = trueDocuements + falseDocuments
    llmModel, graphDocuments = processLLM(combinedData)
    print(f"Combined data completeted for {index}")
    combinedNodes = getNodesListIDs(graphDocuments)
    print(combinedNodes) ## temp
    # result = {
    #     "true": trueNodes,
    #     "false": falseNodes,
    #     "combined": combinedNodes
    # }
    graph = queries.neo4j()
    driver = queries.driveOpen()
    try:
        queries.clearDataWithIndex(driver)
        print(f"Database Cleaned Successfuly.")
    except Exception as e:
         print(f"Data Clean error : {str(e)}")
    finally:
        queries.driveClose(driver)
    #add data to database
    addToGraph(graph, graphDocuments) 
    print(f"Data added to Graph for {index}")
    
    #create index of the database
    driver = queries.driveOpen()
    try:
        queries.createIndex(driver)
        print("Indexing created successfully.")
    except Exception as e:
        print(f"Index creation skipped: {str(e)}")
    finally:
        queries.driveClose(driver)
    print("Data ingestion completed successfully!\n")
    queries.setupGdsCentrality(graph)
    bridges = queries.bridgeNodeConnector(graph, 0.7, 20)
    print(bridges)
    # res = dataGeneratorForLogistic(graph, result)
    # print(f"response genereted for {index}")
    return


if __name__ == "__main__":
    # x = handleDataIngestion()
    # print(x)
    # graph = queries.neo4j()
    # setup_results = queries.setupGdsCentrality(graph)
    # print("Setup completed:", setup_results)
    # bridges = queries.bridgeNodeConnector(graph, 0.7, 20)
    # print(bridges)
    # result_small = {
    #     "true": [ 'Apple','Trump'],
    #     "false": ['Connery','Covid-19'],
    #     "combined":['Abraham Lincoln', 'Administration', 'Ady Barkan', 'America', 'Americans', 'Amy Sherman', 'Andrew Cuomo', 'Andy Nguyen', 'Animal Agriculture', 'Antibodies', 'Apple', 'Apple Commercial', 'Asylum', 'Australia', 'Baltimore', 'Barack Obama', 'Barkan', 'Bernie Sanders', 'Biden', 'Biden Promise Tracker', 'Bill', 'Bill Gates', 'Bill Mccarthy', 'Breast Cancer', 'Burgess Owens', 'California', 'Careers', 'Charles Schumer', 'Chicago', 'Chicago Tribune', 'Children', 'China', 'Chinese', 'Christian Leaders', 'Christian Mcwhirter', 'Church Services', 'Church-Goers', 'Churches', 'Cities', 'Clean Energy Infrastructure', 'Clinicians', 'Communists', 'Community Policing', 'Computer Salesman', 'Congress', 'Connery', 'Contact', 'Contact Tracing', 'Cornell', 'Cornell Law School', 'Coronavirus', 'Coronavirus Pandemic', 'Corporate Income', 'Corrections', 'Corrections And Updates', 'Corridor Highway Project In Cobb And Cherokee Counties', 'Couple', 'Court Staff', 'Covid Vaccine Distribution', 'Covid-19', 'Covid-19 Lockdown Plans', 'Covid-19 Response Efforts', 'Covid-19 Testing', 'Covid-19 Tracing', 'Covid-19 Vaccine', 'Crime', 'Critical Electric Infrastructure', 'Crossfire Hurricane', 'Curve', 'Cyber Attacks', 'Dakota Gruener', 'Dan Brouillette', 'David Menchetti', 'Debate', 'Defense Department', 'Deferred Action For Childhood Arrivals Program', 'Democratic Governors', 'Democratic National Convention', 'Democratic Party', 'Democratic Primary Debate', 'Department Of Energy', 'District Of Columbia', 'Do Not Sell My Info/Cookie Policy', 'Domestic Policy', 'Dominion', 'Dominion Voting Systems', 'Donald', 'Donald J. Trump', 'Donald Trump', 'Donate', 'Donations', 'Dtap', 'Duke (Energy)', 'Ebay', 'Election 2020', 'Electrical Grids', 'Electronic Systems', 'Elias Atienza', 'Ella Lee', 'Empire', 'Enewspaper', 'Entertainment', 'Environment', 'Ethical Principles', 'European Union', 'Facebook', 'Facebook Post', 'Facebook Posts', 'Fact Check', 'Factcheck.Org', 'Fascists', 'Fbi', 'Fda', 'Federal Bureau Of Investigation', 'Federal Emergency Management Agency', 'Federal Funding', 'Federal Government', 'Federal Register', 'Federal Spending', 'Federal Stimulus Money For Schools', 'Feedback', 'Flipboard', 'Florida', 'Follow Us', 'Foreign Adversaries', 'Foreign Policy', 'Foreign Threats', 'Forever', 'Fort Mchenry', 'Fort Mchenry National Monument And Historic Shrine', 'Fossil Fuels', 'Fracking', 'Francis Scott Key', 'Funding', 'Gannett', 'Gas-Powered Cars', 'Gates Connection', 'Gates Foundation', 'Gavi', 'Gavi, The Vaccine Alliance', 'Georgia', 'Germany', 'Global Trafficking Rings', 'Government', 'Governors', 'Govtrack', 'Gowns', 'Green New Deal', 'Greenville City Council', 'Greenville, Mississippi', 'Gross Domestic Product', 'Guns', 'H.R. 6666', 'Health Care', 'Health Centers', 'Health Crisis', 'Help Center', 'Hepatitis A', 'Hepatitis B', 'Her', 'Home Delivery', 'Homeland Security Investigations', 'Horse’S Ass', 'Hospitals', 'House', 'House Committee On Energy And Commerce', 'Houses Of Worship', 'Housing', 'Hpv', 'Hr6666', 'Hugh Hewitt', 'Human Trafficking', 'Human Trafficking Victims', 'Humanitarian Needs', 'Hydraulic Fracturing', 'Id2020', 'Ifcn', 'Ill.', 'Illegal Immigrants', 'Illinois', 'Illinois House Labor And Commerce Committee', 'Immigrants', 'Immigration', 'Immigration And Customs Enforcement', 'Immigration Cases', 'Immigration Judges', 'Immigration Officials', 'Immigration Plan', 'Immunization Action Coalition', 'In-Person Learning', 'Indiana', 'Influenza', 'Infrastructure Bill', 'Instagram', 'Intelligence Community', 'Internet Servers', 'Internships', 'Interpreters', 'Iowa', 'Iran', 'Isolation', 'Italy', 'J. Trump', 'James Bond', 'James Madison', 'Janet Yellen', 'Japan', 'Jeff Foxworthy', 'Joe', 'Joe Biden', 'John Adams', 'John Velazquez', 'John Willshire', 'Jon Greenberg', 'Joni Ernst', 'July 29', 'June', 'June 26', 'June 5, 2008', 'Justice Department', 'Kaiser Health News', 'Kamala', 'Kamala Harris', 'Karen Pence', 'Kasim Reed', 'Kayleigh Mcenany', 'Kentucky Derby', 'Kristi Noem', 'Larry Hogan', 'Law Enforcement', 'Lawful Permanent Residents', 'Lebron James', 'Legal Action', 'Legislatures', 'Licensing & Reprints', 'Life', 'Lincoln', 'Linkedin', 'Liquor Stores', 'Local Business News', 'Local Governments', 'Louis Jacobson', 'Louise Slaughter', 'Madison', 'March 13', 'March 2, 2017', 'Maricopa County', 'Marijuana', 'Marsha Blackburn', 'Maryland', 'May 15', 'May 15 Press Briefing', 'May 2, 2021', 'May 3, 2021', 'May 30, 2021', 'Mayo Clinic', 'Mayor Errick D. Simmons', 'Mayors', 'Mcclatchy D.C.', 'Mcclatchy Dc', 'Meet The Press', 'Melinda Gates', 'Mental Health Counseling', 'Michael Flynn', 'Michigan', 'Microsoft', 'Migrants', 'Mike Pence', 'Military', 'Miriam Valverde', 'Misinformation', 'Mississippi', 'Mississippi Churchgoers', 'Missouri', 'Mitch Mcconnell', 'Mmr', 'Mobile Apps', 'Mobs', 'Money', 'Monthly', 'Mosques', 'My Account', 'N-95 Respirators', 'Nancy Pelosi', 'National Anthem', 'National Conference Of State Legislatures', 'Natural Gas', 'Nbc', 'Net-Zero', 'Net-Zero Emissions', 'New York', 'New York Times', 'News', 'Newsletters', 'Newsroom', 'Newsrooms', 'Nextgen Climate Action Committee', 'Nonprofit Organizations', 'North Carolina', 'Northwest Corridor Highway', 'November', 'November 28, 2016', 'Npr', 'Obama', 'Obama-Biden Administration', 'Obameter', 'Officials Dispute Trump’S Claim', 'Oil', 'One Time', 'Online Prayers', 'Open Borders', 'Opinion', 'Organization', 'Os X', 'Our Process', 'Our Staff', 'Pandemic', 'Pants On Fire', 'Patients', 'Paul Manafort', 'Paul Specht', 'Pence', 'Pennsylvania', 'People', 'Personal Protective Equipment', 'Pinterest', 'Places Of Worship', 'Plaintiffs', 'Plasma', 'Podcasts', 'Police', 'Police Funding', 'Polio', 'Politifact', 'Politifact Staff', 'Poynter Institute', 'Preexisting Conditions', 'President', 'Press Releases', 'Privacy Policy', 'Promise Tracker', 'Public Health Officials', 'Quarantine', 'Rachel Maddow', 'Randy Feenstra', 'Readers', 'Recent Articles And Fact-Checks', 'Recent Fact-Checks', 'Reddit', 'Reform', 'Religious And Philosophical Exemptions', 'Remarks By President Trump On Vaccine Development', 'Republican Convention', 'Republican National Committee', 'Republican National Convention', 'Republican Party Nomination', 'Republicans', 'Resolution', 'Respirators', 'Richard Grinell', 'Rick Barr', 'Rick Scott', 'Roads & Bridges', 'Ron Johnson', 'Rss', 'Rss Feeds', 'Rush Limbaugh', 'Russia', 'Samantha Putterman', 'Sanctuary Cities', 'Satellite Information Network, Llc', 'School', 'School Choice', 'School Immunization Laws', 'School Immunization Requirements', 'Schools', 'Scoopertino', 'Sean Connery', 'Sean Hannity', 'Senate', 'September 10, 2017', 'Shop', 'Sitemap', 'Small Businesses', 'Social', 'Social Distancing', 'Social Media Platforms', 'Socialism', 'Son', 'Soviet Union', 'Sports', 'Sports Weekly Studio', 'St. Petersburg, Fl', 'Staff', 'State', 'State Governments', 'State Laws And Mandates By Vaccine', 'State School Immunization', 'State Vaccination Requirements', 'States', 'Steve Jobs', 'Stimulus Money', 'Stores And Abortion Clinics', 'Storytellers', 'Submitting Letters To The Editor', 'Subscribe', 'Suggest A Fact-Check', 'Supplies', 'Support', 'Synagogue', 'Tax Deductible Contribution', 'Taxes', 'Tech', 'Tech Bias Story Sharing Tool', 'Terms & Conditions', 'Terms Of Service', 'Testing', 'Texas', 'The City', 'The Daily Beast', 'The Facts Newsletter', 'The Poynter Institute Menu', 'The Star-Spangled Banner', 'The Wall', 'Thomas Jefferson', 'Tiktok', 'Tips', 'Trace Act', 'Trafficking', 'Trans-Pacific Partnership', 'Travel', 'Travel Restrictions', 'Treatment_1', 'Treatment_2', 'Trump', 'Trump 2016 Campaign', 'Trump Administration', 'Trump Campaign', 'Trump White House', 'Trump-O-Meter', 'Trump’S Presidency', 'Trusted Information', 'Truth', 'Truth Squad', 'Truth-O-Meter', 'Tucker Carlson', 'Tweet', 'Twitter', 'U.S.', 'U.S. Citizens', 'U.S. Government', 'United States', 'United States Bulk-Power System', 'United States’ National Anthem', 'Us Trafficking Arrests', 'Usa Today', 'User Policies', 'Utah', 'Vaccinations', 'Vaccine', 'Vaccines', 'Ventilators', 'Vermont', 'Victoria Knight', 'Virginia', 'Virus', 'Washington', 'Washington, D.C.', 'Washington, Dc', 'Wealthy', 'West Virginia', 'White House', 'Who', 'Who Meeting', 'William Barr', 'Willie Wilson', 'Willshire', 'Wisconsin', 'Wisconsin Schools', 'World Health Organization', 'Wreg', 'Wreg Memphis', 'Wuhan', 'Xi Jinping', 'Yearly', 'You', 'Young Men’S Lyceum Of Springfield', 'Your California Privacy Rights/Privacy Policy', 'Youtube']
    # }
    # result = {
    #     "true": [ 'Apple', 'Apple Commercial', 'Connery', 'Democratic Governors', 'Dominion', 'Donald Trump', 'Elias Atienza', 'Jeff Foxworthy', 'John Adams', 'John Willshire', 'Sean Connery', 'Steve Jobs', 'Thomas Jefferson', 'Twitter', 'Willshire'],
    #     "false": ['Abraham Lincoln','Ady Barkan','Amy Sherman','Andrew Cuomo','Barack Obama','Bernie Sanders','Biden','Bill Gates','Burgess Owens','California','Charles Schumer','Chicago','China','Congress','Coronavirus','Dan Brouillette','Donald Trump','Kamala Harris','Nancy Pelosi','Xi Jinping'],
    #     "combined":['Abraham Lincoln', 'Administration', 'Ady Barkan', 'America', 'Americans', 'Amy Sherman', 'Andrew Cuomo', 'Andy Nguyen', 'Animal Agriculture', 'Antibodies', 'Apple', 'Apple Commercial', 'Asylum', 'Australia', 'Baltimore', 'Barack Obama', 'Barkan', 'Bernie Sanders', 'Biden', 'Biden Promise Tracker', 'Bill', 'Bill Gates', 'Bill Mccarthy', 'Breast Cancer', 'Burgess Owens', 'California', 'Careers', 'Charles Schumer', 'Chicago', 'Chicago Tribune', 'Children', 'China', 'Chinese', 'Christian Leaders', 'Christian Mcwhirter', 'Church Services', 'Church-Goers', 'Churches', 'Cities', 'Clean Energy Infrastructure', 'Clinicians', 'Communists', 'Community Policing', 'Computer Salesman', 'Congress', 'Connery', 'Contact', 'Contact Tracing', 'Cornell', 'Cornell Law School', 'Coronavirus', 'Coronavirus Pandemic', 'Corporate Income', 'Corrections', 'Corrections And Updates', 'Corridor Highway Project In Cobb And Cherokee Counties', 'Couple', 'Court Staff', 'Covid Vaccine Distribution', 'Covid-19', 'Covid-19 Lockdown Plans', 'Covid-19 Response Efforts', 'Covid-19 Testing', 'Covid-19 Tracing', 'Covid-19 Vaccine', 'Crime', 'Critical Electric Infrastructure', 'Crossfire Hurricane', 'Curve', 'Cyber Attacks', 'Dakota Gruener', 'Dan Brouillette', 'David Menchetti', 'Debate', 'Defense Department', 'Deferred Action For Childhood Arrivals Program', 'Democratic Governors', 'Democratic National Convention', 'Democratic Party', 'Democratic Primary Debate', 'Department Of Energy', 'District Of Columbia', 'Do Not Sell My Info/Cookie Policy', 'Domestic Policy', 'Dominion', 'Dominion Voting Systems', 'Donald', 'Donald J. Trump', 'Donald Trump', 'Donate', 'Donations', 'Dtap', 'Duke (Energy)', 'Ebay', 'Election 2020', 'Electrical Grids', 'Electronic Systems', 'Elias Atienza', 'Ella Lee', 'Empire', 'Enewspaper', 'Entertainment', 'Environment', 'Ethical Principles', 'European Union', 'Facebook', 'Facebook Post', 'Facebook Posts', 'Fact Check', 'Factcheck.Org', 'Fascists', 'Fbi', 'Fda', 'Federal Bureau Of Investigation', 'Federal Emergency Management Agency', 'Federal Funding', 'Federal Government', 'Federal Register', 'Federal Spending', 'Federal Stimulus Money For Schools', 'Feedback', 'Flipboard', 'Florida', 'Follow Us', 'Foreign Adversaries', 'Foreign Policy', 'Foreign Threats', 'Forever', 'Fort Mchenry', 'Fort Mchenry National Monument And Historic Shrine', 'Fossil Fuels', 'Fracking', 'Francis Scott Key', 'Funding', 'Gannett', 'Gas-Powered Cars', 'Gates Connection', 'Gates Foundation', 'Gavi', 'Gavi, The Vaccine Alliance', 'Georgia', 'Germany', 'Global Trafficking Rings', 'Government', 'Governors', 'Govtrack', 'Gowns', 'Green New Deal', 'Greenville City Council', 'Greenville, Mississippi', 'Gross Domestic Product', 'Guns', 'H.R. 6666', 'Health Care', 'Health Centers', 'Health Crisis', 'Help Center', 'Hepatitis A', 'Hepatitis B', 'Her', 'Home Delivery', 'Homeland Security Investigations', 'Horse’S Ass', 'Hospitals', 'House', 'House Committee On Energy And Commerce', 'Houses Of Worship', 'Housing', 'Hpv', 'Hr6666', 'Hugh Hewitt', 'Human Trafficking', 'Human Trafficking Victims', 'Humanitarian Needs', 'Hydraulic Fracturing', 'Id2020', 'Ifcn', 'Ill.', 'Illegal Immigrants', 'Illinois', 'Illinois House Labor And Commerce Committee', 'Immigrants', 'Immigration', 'Immigration And Customs Enforcement', 'Immigration Cases', 'Immigration Judges', 'Immigration Officials', 'Immigration Plan', 'Immunization Action Coalition', 'In-Person Learning', 'Indiana', 'Influenza', 'Infrastructure Bill', 'Instagram', 'Intelligence Community', 'Internet Servers', 'Internships', 'Interpreters', 'Iowa', 'Iran', 'Isolation', 'Italy', 'J. Trump', 'James Bond', 'James Madison', 'Janet Yellen', 'Japan', 'Jeff Foxworthy', 'Joe', 'Joe Biden', 'John Adams', 'John Velazquez', 'John Willshire', 'Jon Greenberg', 'Joni Ernst', 'July 29', 'June', 'June 26', 'June 5, 2008', 'Justice Department', 'Kaiser Health News', 'Kamala', 'Kamala Harris', 'Karen Pence', 'Kasim Reed', 'Kayleigh Mcenany', 'Kentucky Derby', 'Kristi Noem', 'Larry Hogan', 'Law Enforcement', 'Lawful Permanent Residents', 'Lebron James', 'Legal Action', 'Legislatures', 'Licensing & Reprints', 'Life', 'Lincoln', 'Linkedin', 'Liquor Stores', 'Local Business News', 'Local Governments', 'Louis Jacobson', 'Louise Slaughter', 'Madison', 'March 13', 'March 2, 2017', 'Maricopa County', 'Marijuana', 'Marsha Blackburn', 'Maryland', 'May 15', 'May 15 Press Briefing', 'May 2, 2021', 'May 3, 2021', 'May 30, 2021', 'Mayo Clinic', 'Mayor Errick D. Simmons', 'Mayors', 'Mcclatchy D.C.', 'Mcclatchy Dc', 'Meet The Press', 'Melinda Gates', 'Mental Health Counseling', 'Michael Flynn', 'Michigan', 'Microsoft', 'Migrants', 'Mike Pence', 'Military', 'Miriam Valverde', 'Misinformation', 'Mississippi', 'Mississippi Churchgoers', 'Missouri', 'Mitch Mcconnell', 'Mmr', 'Mobile Apps', 'Mobs', 'Money', 'Monthly', 'Mosques', 'My Account', 'N-95 Respirators', 'Nancy Pelosi', 'National Anthem', 'National Conference Of State Legislatures', 'Natural Gas', 'Nbc', 'Net-Zero', 'Net-Zero Emissions', 'New York', 'New York Times', 'News', 'Newsletters', 'Newsroom', 'Newsrooms', 'Nextgen Climate Action Committee', 'Nonprofit Organizations', 'North Carolina', 'Northwest Corridor Highway', 'November', 'November 28, 2016', 'Npr', 'Obama', 'Obama-Biden Administration', 'Obameter', 'Officials Dispute Trump’S Claim', 'Oil', 'One Time', 'Online Prayers', 'Open Borders', 'Opinion', 'Organization', 'Os X', 'Our Process', 'Our Staff', 'Pandemic', 'Pants On Fire', 'Patients', 'Paul Manafort', 'Paul Specht', 'Pence', 'Pennsylvania', 'People', 'Personal Protective Equipment', 'Pinterest', 'Places Of Worship', 'Plaintiffs', 'Plasma', 'Podcasts', 'Police', 'Police Funding', 'Polio', 'Politifact', 'Politifact Staff', 'Poynter Institute', 'Preexisting Conditions', 'President', 'Press Releases', 'Privacy Policy', 'Promise Tracker', 'Public Health Officials', 'Quarantine', 'Rachel Maddow', 'Randy Feenstra', 'Readers', 'Recent Articles And Fact-Checks', 'Recent Fact-Checks', 'Reddit', 'Reform', 'Religious And Philosophical Exemptions', 'Remarks By President Trump On Vaccine Development', 'Republican Convention', 'Republican National Committee', 'Republican National Convention', 'Republican Party Nomination', 'Republicans', 'Resolution', 'Respirators', 'Richard Grinell', 'Rick Barr', 'Rick Scott', 'Roads & Bridges', 'Ron Johnson', 'Rss', 'Rss Feeds', 'Rush Limbaugh', 'Russia', 'Samantha Putterman', 'Sanctuary Cities', 'Satellite Information Network, Llc', 'School', 'School Choice', 'School Immunization Laws', 'School Immunization Requirements', 'Schools', 'Scoopertino', 'Sean Connery', 'Sean Hannity', 'Senate', 'September 10, 2017', 'Shop', 'Sitemap', 'Small Businesses', 'Social', 'Social Distancing', 'Social Media Platforms', 'Socialism', 'Son', 'Soviet Union', 'Sports', 'Sports Weekly Studio', 'St. Petersburg, Fl', 'Staff', 'State', 'State Governments', 'State Laws And Mandates By Vaccine', 'State School Immunization', 'State Vaccination Requirements', 'States', 'Steve Jobs', 'Stimulus Money', 'Stores And Abortion Clinics', 'Storytellers', 'Submitting Letters To The Editor', 'Subscribe', 'Suggest A Fact-Check', 'Supplies', 'Support', 'Synagogue', 'Tax Deductible Contribution', 'Taxes', 'Tech', 'Tech Bias Story Sharing Tool', 'Terms & Conditions', 'Terms Of Service', 'Testing', 'Texas', 'The City', 'The Daily Beast', 'The Facts Newsletter', 'The Poynter Institute Menu', 'The Star-Spangled Banner', 'The Wall', 'Thomas Jefferson', 'Tiktok', 'Tips', 'Trace Act', 'Trafficking', 'Trans-Pacific Partnership', 'Travel', 'Travel Restrictions', 'Treatment_1', 'Treatment_2', 'Trump', 'Trump 2016 Campaign', 'Trump Administration', 'Trump Campaign', 'Trump White House', 'Trump-O-Meter', 'Trump’S Presidency', 'Trusted Information', 'Truth', 'Truth Squad', 'Truth-O-Meter', 'Tucker Carlson', 'Tweet', 'Twitter', 'U.S.', 'U.S. Citizens', 'U.S. Government', 'United States', 'United States Bulk-Power System', 'United States’ National Anthem', 'Us Trafficking Arrests', 'Usa Today', 'User Policies', 'Utah', 'Vaccinations', 'Vaccine', 'Vaccines', 'Ventilators', 'Vermont', 'Victoria Knight', 'Virginia', 'Virus', 'Washington', 'Washington, D.C.', 'Washington, Dc', 'Wealthy', 'West Virginia', 'White House', 'Who', 'Who Meeting', 'William Barr', 'Willie Wilson', 'Willshire', 'Wisconsin', 'Wisconsin Schools', 'World Health Organization', 'Wreg', 'Wreg Memphis', 'Wuhan', 'Xi Jinping', 'Yearly', 'You', 'Young Men’S Lyceum Of Springfield', 'Your California Privacy Rights/Privacy Policy', 'Youtube']
    # }
    # x = dataGeneratorForLogistic(graph, result_small)
    # print(x)
    handleDataIngestion()
    print("Session terminated.")