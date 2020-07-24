"""add language to posts and the news articles

Revision ID: bceb531e5734
Revises: a0acc7a608ab
Create Date: 2020-04-16 17:06:09.374237

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bceb531e5734'
down_revision = 'a0acc7a608ab'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('news', sa.Column('language', sa.String(length=5), nullable=True))
    op.add_column('post', sa.Column('language', sa.String(length=5), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('post', 'language')
    op.drop_column('news', 'language')
    # ### end Alembic commands ###