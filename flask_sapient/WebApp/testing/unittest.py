# The unittest class is used when developing the webapp, our project is about using the back-end to analyse the video
# Therefore for most cases, the unit testing is hard to be done by code under the theory of TDD. The reason is that,
# the video output have to be run and checked by human to see the exactly the AI information outline on the video
# is correct which is not static at all, it changes every time by different cases
#
# Therefore when we are doing the development, most TDD is done by hand other than code. These unit testing is
# just used to test the whole program can catch up the exception, or can run the video successfully.
# For most functions tested in the unit test is now been disabled for real application, the reason is that
# it is not worth to keep unit tests which cannot cover most cases, everything still need check by hand.

# Since the camera.py and video.py work the same, so we only did the video.py to test the performance

import flaskr
import unittest
import tempfile

from ..Face_Item.video import gen as face_item_video_reading


class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.db_fd, flaskr.app.config['DATABASE'] = tempfile.mkstemp()
        flaskr.app.testing = True
        self.app = flaskr.app.test_client()
        with flaskr.app.app_context():
            flaskr.init_db()

    # first test to test empty databse
    def test_empty_db(self):
        rv = self.app.get('/')
        assert b'No entries here so far' in rv.data

    # invalid login
    def test_login_logout(self):
        rv = self.login('admin', 'pass')
        assert b'You were logged in' in rv.data
        rv = self.logout()
        assert b'You were logged out' in rv.data
        rv = self.login('admin', 'default')
        assert b'Invalid username' in rv.data

    # to check the back-end app read the video succefully
    def test_successfully_run(self):
        self.assertEqual(face_item_video_reading, "sucess")

    # to check the back-end app catch up the exception successfully
    def test_isupper(self):
        self.assertEqual(face_item_video_reading, "exception raised")


if __name__ == '__main__':
    unittest.main()