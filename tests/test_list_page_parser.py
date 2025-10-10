import os
import sys
import types


PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))
if "professor_analysis" not in sys.modules:
        package = types.ModuleType("professor_analysis")
        package.__path__ = [PACKAGE_ROOT]
        sys.modules["professor_analysis"] = package


from professor_analysis.list_page_parser import extract_profiles_from_listing


def test_unm_detector_extracts_profiles():
        html = """
        <div class="main-content">
            <div class="col-sm-12">
                <h3>Jane Doe</h3>
                <h4>Associate Professor</h4>
                <p>Email: jane.doe@example.edu</p>
            </div>
        </div>
        """

        profiles = extract_profiles_from_listing(html, "https://example.com/listing", None, "Communication")

        assert profiles
        assert profiles[0].professor_name == "Jane Doe"
        assert profiles[0].contact_info.emails == ["jane.doe@example.edu"]


def test_uky_detector_extracts_profiles():
        html = """
        <main>
            <h3>Department of Communication</h3>
            <p><strong>John Smith</strong> Assistant Professor</p>
        </main>
        """

        profiles = extract_profiles_from_listing(html, "https://example.com/uky", None, None)

        assert profiles
        assert profiles[0].professor_name == "John Smith"
        assert profiles[0].program_name == "Department of Communication"


def test_lsu_detector_extracts_profiles():
        html = """
        <table class="scholarships">
            <tr>
                <td>
                    <span style="font-size: 18px;"><a href="/faculty/jane">Jane Tiger</a></span>
                    <div>Professor of Practice</div>
                </td>
            </tr>
        </table>
        """

        profiles = extract_profiles_from_listing(html, "https://example.com/lsu", None, None)

        assert profiles
        assert profiles[0].professor_name == "Jane Tiger"
        assert profiles[0].university_name == "Louisiana State University"


def test_syracuse_detector_extracts_profiles():
        html = """
        <section data-component="faculty-card">
            <div>
                <div>Jane Orange</div>
                <div>Assistant Professor</div>
                <div>Communications</div>
                <a href="/faculty/jane">View Bio</a>
            </div>
        </section>
        """

        profiles = extract_profiles_from_listing(html, "https://example.com/syracuse", None, None)

        assert profiles
        assert profiles[0].professor_name == "Jane Orange"
        assert "syracuse" in profiles[0].university_name.lower()


def test_udel_detector_extracts_profiles():
        html = """
        <ul>
            <li>
                <img alt="Headshot photo of Blue Hen" src="hen.jpg" />
                <strong>Chair</strong>
                <a href="mailto:blue.hen@example.edu">blue.hen@example.edu</a>
            </li>
        </ul>
        """

        profiles = extract_profiles_from_listing(html, "https://example.com/udel", None, None)

        assert profiles
        assert profiles[0].professor_name == "Blue Hen"
        assert profiles[0].contact_info.emails == ["blue.hen@example.edu"]


def test_asu_detector_extracts_profiles():
        html = """
        <div id="clas-isearch-react" data-config='{"type": "customList", "ids": ["john.doe"]}'></div>
        """

        profiles = extract_profiles_from_listing(html, "https://example.com/asu", "Arizona State University", "Human Communication")

        assert profiles
        assert profiles[0].profile_url.endswith("/john.doe")
