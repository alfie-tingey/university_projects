"""test

Revision ID: 13d057aa8c3c
Revises: c9154def00cc
Create Date: 2020-04-08 18:25:23.758968

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '13d057aa8c3c'
down_revision = 'c9154def00cc'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    # op.add_column('news', sa.Column('id', sa.Integer(), nullable=True))
    # ### end Alembic commands ###
    pass


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    # op.drop_column('news', 'id')
    # ### end Alembic commands ###
    pass
