"""Tests for website generation.

"""

import os
from glob import glob
from datetime import datetime

import pytest
from bs4 import BeautifulSoup

from fah_xchem.schema import TimestampedAnalysis
from fah_xchem.analysis.website import WebsiteArtifactory
from fah_xchem.data import get_compound_series_analysis_results


@pytest.fixture
def website_artifactory(tmpdir):

    with tmpdir.as_cwd():
        compound_series_analysis = get_compound_series_analysis_results()
        waf = WebsiteArtifactory(
                base_url="https://localhost:8080",
                path=os.getcwd(),
                series=compound_series_analysis,
                timestamp=datetime(2021, 7, 12, 0, 0, 0),
                fah_ws_api_url=None,
                )

    return waf


class TestWebsiteArtifactory:

    page_titles = (
            "Summary",
            "Compounds",
            "Microstates",
            "Transformations",
            "Reliable Transformations",
            "Retrospective Transformations"
            )


    def test_init(self, website_artifactory):
        waf = website_artifactory

        assert waf.environment

    def test_generate_website(self):
        """Test overall website generation.

        """
        ...

    def _test_navbar(self, body):
        navbar = body.find_all('div', 'navbar-nav')[0]

        # ensure all page titles present among items, in order
        for item, page_title in zip(navbar.find_all('a', ['nav-item', 'nav-link']),
                                    self.page_titles):
            assert page_title in item.contents[0]
    
    def test_generate_summary(self, website_artifactory):
        """Test Summary page content generation.

        """
        waf = website_artifactory
        num_top_compounds = 5

        waf.generate_summary(num_top_compounds)

        # introspect generated page
        with open(os.path.join(waf.path, "index.html"), 'r') as f:
            soup = BeautifulSoup(f)

        # check key elements of summary page
        body = soup.body

        # navbar
        self._test_navbar(body)

        # number of top compounds displayed
        ## number of rows in table includes header
        assert len(body.table.find_all('tr')) == num_top_compounds + 1

        # two plots 
        desc_head = body.find_all('h3', text='Distributions')[0]

        # relative free energy distribution
        rel_filename = 'relative_fe_dist'
        rel = desc_head.find_next_sibling('a')

        assert rel['href'] == rel_filename + ".pdf"
        assert rel.img['src'] == rel_filename + ".png"

        # cumulative free energy distribution
        cum_filename = 'cumulative_fe_dist'
        cum = rel.find_next_sibling('a')

        assert cum['href'] == cum_filename + ".pdf"
        assert cum.img['src'] == cum_filename + ".png"

        # raw data link for analysis.json
        filename = 'analysis.json'
        raw_data_head = body.find_all('h3', text='Raw data')[0]
        file_list = raw_data_head.find_next_sibling('ul')

        assert len(file_list.contents) == 1
        item = file_list.li
        assert item.a['href'] == filename
        assert item.a.contents[0] == filename

        # PDF summaries
        pdf_summary_head = body.find_all('h4', text='PDF summary')[0]
        file_list = pdf_summary_head.find_next_sibling('ul')

        assert len(file_list.find_all('li')) == 2
        filename = "transformations-final-ligands.pdf"
        for filename, item in zip(
                ("transformations-final-ligands.pdf", 
                 "reliable-transformations-final-ligands.pdf"),
                file_list.find_all('li')):
            assert item.a['href'] == filename
            assert item.a.contents[0] == filename
        
        # structures
        structures = body.find_all('h3', text='Structures')[0]

        ## proposed ligands
        proposed_ligands = body.find_all('h4', text='Proposed ligands')[0]
        file_list = proposed_ligands.find_next_sibling('ul')

        filename = "transformations-final-ligands"
        for ftype, item in zip(('csv', 'sdf', 'mol2'), file_list.find_all('li')):
            assert item.a['href'] == f"{filename}.{ftype}"
            assert item.a.contents[0] == f"{filename}.{ftype}"

        item = file_list.find_all('li')[-1]
        assert item.a['href'] == 'transformations-final-proteins.pdb'
        assert item.a.contents[0] == 'transformations-final-proteins.pdb'

        ### proposed ligands with reliable transformations
        proposed_ligands_rel_trans = body.find_all('h5',
                                                   text='Proposed ligands with reliable transformations')[0]
        file_list = proposed_ligands_rel_trans.find_next_sibling('ul')

        filename = "reliable-transformations-final-ligands"
        for ftype, item in zip(('csv', 'sdf', 'mol2'), file_list.find_all('li')):
            assert item.a['href'] == f"{filename}.{ftype}"
            assert item.a.contents[0] == f"{filename}.{ftype}"

        item = file_list.find_all('li')[-1]
        assert item.a['href'] == 'reliable-transformations-final-proteins.pdb'
        assert item.a.contents[0] == 'reliable-transformations-final-proteins.pdb'

        ## reference ligands
        reference_ligands = body.find_all('h4', text='Reference ligands')[0]
        file_list = reference_ligands.find_next_sibling('ul')

        filename = "transformations-initial-ligands"
        for ftype, item in zip(('sdf', 'mol2'), file_list.find_all('li')):
            assert item.a['href'] == f"{filename}.{ftype}"
            assert item.a.contents[0] == f"{filename}.{ftype}"

        ### reference ligands with reliable transformations
        reference_ligands_rel_trans = body.find_all('h5',
                                                    text='Reference ligands with reliable transformations')[0]
        file_list = reference_ligands_rel_trans.find_next_sibling('ul')

        filename = "reliable-transformations-initial-ligands"
        for ftype, item in zip(('sdf', 'mol2'), file_list.find_all('li')):
            assert item.a['href'] == f"{filename}.{ftype}"
            assert item.a.contents[0] == f"{filename}.{ftype}"


    def test_generate_compounds(self, website_artifactory):
        """Test Compounds page content generation.

        The Compounds page generation also includes a details page for the top
        compounds, up to 

        """
        waf = website_artifactory
        items_per_page = 7
        num_top_compounds = 5

        waf.generate_compounds(items_per_page=items_per_page,
                               num_top_compounds=num_top_compounds)
        
        # grab up all generated index pages
        index_pages = sorted(glob(os.path.join(waf.path, "compounds", "index*.html")))

        # put `index.html` at the front; sorting puts it at the back
        index_pages.insert(0, index_pages.pop(-1))

        # our test dataset should give two pages
        assert len(index_pages) == 2

        for index_page in index_pages:
            page_name = os.path.basename(index_page)

            # introspect generated page
            with open(index_page, 'r') as f:
                soup = BeautifulSoup(f)

            # check key elements of page
            body = soup.body
            compounds_head = body.find_all("h3", text="Compounds")[0]

            # check pagination and links
            pagination = compounds_head.find_next_sibling('div', "my-3")
            if page_name == "index.html":
                assert pagination.contents[0].strip().startswith(f"Showing 1 through {items_per_page}")
                assert len(pagination.find_all('a')) == 1
                assert pagination.find_all('a')[0]['href'].startswith(f"compounds/index-{items_per_page+1}")
            else:
                _, start, stop = page_name.split('.')[0].split('-')
                assert pagination.contents[0].strip().startswith(f"Showing {start} through {stop}")

                # if we're looking at the last index page
                if index_page == index_pages[-1]:
                    assert len(pagination.find_all('a')) == 1
                    assert pagination.find_all('a')[0]
                else:
                    assert len(pagination.find_all('a')) == 2

            # check count of rows against count of compounds with free energy data
            ## TODO: we won't have a row for a compound without data; perhaps we should make them though?
            
            # check count of items in table isn't beyond `items_per_page`
            contents = pagination.contents[0].strip().split()
            start, stop = int(contents[1]), int(contents[3])

            # all table rows, includes header
            items = compounds_head.find_next_sibling('table', ['table', 'table-striped']).find_all('tr')
            assert len(items) - 1 == (stop - start) + 1

            # check content expectations for each item in table
            for item in items:
                if item.th:
                    # this is the header row
                    assert len(item.find_all('th')) == 5
                else:
                    # compound rows have a thumbnail column that has no label
                    # 1 more column than header
                    assert len(item.find_all('td')) == 6


            # FIXME: second page top compounds get a broken link


    def test_top_compounds_detail(self, website_artifactory):
        waf = website_artifactory
        items_per_page = 7
        num_top_compounds = 5

        waf.generate_compounds(items_per_page=items_per_page,
                               num_top_compounds=num_top_compounds)
        
        # grab up main index page
        index_page = os.path.join(waf.path, "compounds", "index.html")

        # introspect generated page
        with open(index_page, 'r') as f:
            soup = BeautifulSoup(f)
        

    def test_generate_microstates(self):
        """Test Microstates page content generation.

        """
        ...

    def test_generate_transformations(self):
        """Test Transformations page content generation.

        """
        ...

    def test_generate_reliable_transformations(self):
        """Test Reliable Transformations page content generation.

        """
        ...

    def test_generate_retrospective_transformations(self):
        """Test Retrspective Transformations page content generation.

        """
        ...
