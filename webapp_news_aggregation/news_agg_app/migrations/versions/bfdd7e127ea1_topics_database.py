"""topics_database

Revision ID: bfdd7e127ea1
Revises: 9d09e3f4eb21
Create Date: 2020-04-29 22:06:59.792446

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bfdd7e127ea1'
down_revision = '9d09e3f4eb21'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('topics_database',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('date', sa.String(length=140), nullable=True),
    sa.Column('category', sa.String(length=140), nullable=True),
    sa.Column('topic', sa.String(length=140), nullable=False),
    sa.Column('score', sa.String(length=140), nullable=True),
    sa.Column('language', sa.String(length=5), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('topics_database')
    # ### end Alembic commands ###